import torch
import flashinfer
from config import S_dec, D, N, H
import model
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

def measure_decode_isolated(B):
  # Setting: Decodes using one green context; remaining SMs outside context idle 

  # Set up KV caches on GPU HBM
  k_cache = torch.randn((B, S_dec, N, H), dtype = torch.float16, device = "cuda")
  v_cache = torch.randn((B, S_dec, N, H), dtype = torch.float16, device = "cuda")
  paged_kv = torch.stack([k_cache, v_cache], dim = 1)
 
  activation = torch.randn((B, 1, D), dtype = torch.float16, device = "cuda")

  # Set up paging config of KV cache
  workspace = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

  kv_indptr = torch.arange(0, B + 1, dtype = torch.int32, device = "cuda")
  kv_indices = torch.arange(0, B, dtype = torch.int32, device = "cuda")
  kv_last_page_len = torch.full((B, ), S_dec, dtype = torch.int32, device = "cuda")

  # Set up FlashInfer wrapper object for decode

  decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads = N, num_kv_heads = N, head_dim = H, page_size = S_dec)


  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  dev = torch.device("cuda:0")
  granularity = 8

  NUM_WARMUPS = 10
  NUM_ITERS= 100 
  no_contention_results = []

  # Without any partition - decodes have all SMs
  # Warmup: launch some decode kernel
  with torch.inference_mode():
    for _ in range(NUM_WARMUPS):
      out = model.do_batched_decode(activation, paged_kv, decode_wrapper)

  start = torch.cuda.Event(enable_timing = True)
  end = torch.cuda.Event(enable_timing = True)

  torch.cuda.synchronize()

  start.record()

  # Actual decode runs for timing
  with torch.inference_mode():
    for _ in range(NUM_ITERS):
      out = model.do_batched_decode(activation, paged_kv, decode_wrapper)

  end.record()

  torch.cuda.synchronize()

  itl = start.elapsed_time(end) / NUM_ITERS

  print(f"Without contention (Batch Size: {B}, Active SMs: {num_sms}) inter_token_latency (ms): {itl:.3f}")

  no_contention_results.append({"Batch": B, "Active_SMs": num_sms, "Elapsed_time_ms": round(itl, 3)})


  # Sweep over partition configs. Create green context + associated stream for each 
  for i in range(1, num_sms // granularity):

    active_sms = num_sms - i*granularity
    streams, resources = split_device_green_ctx_by_sm_count(dev, [active_sms])

    target_stream = streams[0]
    with torch.cuda.stream(target_stream):

      # Warmup: launch some decode kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS):
          out = model.do_batched_decode(activation, paged_kv, decode_wrapper)

      start = torch.cuda.Event(enable_timing = True)
      end = torch.cuda.Event(enable_timing = True)

      target_stream.synchronize()

      start.record()

      # Actual decode runs for timing
      with torch.inference_mode():
        for _ in range(NUM_ITERS):
          out = model.do_batched_decode(activation, paged_kv, decode_wrapper)

      end.record()

      target_stream.synchronize()

      itl = start.elapsed_time(end) / NUM_ITERS

      print(f"Without contention (Batch Size: {B}, Active SMs: {active_sms}) inter_token_latency (ms): {itl:.3f}")

      no_contention_results.append({"Batch": B, "Active_SMs": active_sms, "Elapsed_time_ms": round(itl, 3)})

  return no_contention_results

def measure_decode_under_prefill_contention(prefill_fn, B_dec, B_prefill, S_prefill):
  # Experiment: Profile decodes with serial or batched prefill running in the other green context 

  # Set up KV caches on GPU HBM
  k_cache = torch.randn((B_dec, S_dec, N, H), dtype = torch.float16, device = "cuda")
  v_cache = torch.randn((B_dec, S_dec, N, H), dtype = torch.float16, device = "cuda")
  paged_kv = torch.stack([k_cache, v_cache], dim = 1)
 
  activation_decode = torch.randn((B_dec, 1, D), dtype = torch.float16, device = "cuda")

  # prefil for one request saturates compute
  activation_prefill = torch.randn((B_prefill, S_prefill, D), dtype = torch.float16, device = "cuda") 

  # Set up FlashInfer wrapper object for decode

  # Set up paging config of KV cache
  workspace_dec = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_dec, "NHD")

  kv_indptr = torch.arange(0, B_dec + 1, dtype = torch.int32, device = "cuda")
  kv_indices = torch.arange(0, B_dec, dtype = torch.int32, device = "cuda")
  kv_last_page_len = torch.full((B_dec, ), S_dec, dtype = torch.int32, device = "cuda")

  decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads = N, num_kv_heads = N, head_dim = H, page_size = S_dec)

  # Set up FlashInfer wrapper object for prefill
  workspace_prefill = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_prefill, "NHD")
  qo_indptr = torch.arange(0, (B_prefill + 1) * S_prefill, S_prefill, dtype=torch.int32, device="cuda:0")
  kv_indptr = qo_indptr.clone()
  prefill_wrapper.plan(qo_indptr, kv_indptr, num_qo_heads = N, num_kv_heads = N, head_dim_qk = H, causal=True)

  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  dev = torch.device("cuda:0")
  granularity = 8

  NUM_WARMUPS = 10 # for decodes
  NUM_ITERS = 100  # for decodes
  contention_decode_results = []

  # Sweep over partition configs. Create green context + associated stream for each 
  print(f"With serial-prefill contention")

  for i in range(1, num_sms // granularity):

    active_sms = num_sms - i*granularity
    # Create green contexts
    streams, resources = split_device_green_ctx_by_sm_count(dev, [active_sms])

    stream_dec = streams[0] 
    stream_prefill = streams[1] 

    with torch.cuda.stream(stream_dec):
      # Warmup: launch some decode kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS):
          out = model.do_batched_decode(activation_decode, paged_kv, decode_wrapper)

    with torch.cuda.stream(stream_prefill):
      # Warmup: launch some prefill kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS // 2):
          if prefill_fn is model.do_batched_prefill:
            out = prefill_fn(activation_prefill, prefill_wrapper)
          else:
            out = prefill_fn(activation_prefill)

    prefill_iters = 2
    
    for _ in range(8):
      start1 = torch.cuda.Event(enable_timing = True)
      end1 = torch.cuda.Event(enable_timing = True)
      start2 = torch.cuda.Event(enable_timing = True)
      end2 = torch.cuda.Event(enable_timing = True)

      torch.cuda.synchronize()

      start2.record(stream_prefill)

      torch.cuda.nvtx.range_push("prefill")
      # inverting order so that prefills are underway before
      # decode kernels start executing from other stream
      with torch.cuda.stream(stream_prefill):
        with torch.inference_mode():
          for _ in range(prefill_iters):
            if prefill_fn is model.do_batched_prefill:
              out = prefill_fn(activation_prefill, prefill_wrapper)
            else:
              out = prefill_fn(activation_prefill)
      torch.cuda.nvtx.range_pop()

      start1.record(stream_dec)

      torch.cuda.nvtx.range_push("decode")
      with torch.cuda.stream(stream_dec):
        with torch.inference_mode():
          for _ in range(NUM_ITERS):
            out = model.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
      torch.cuda.nvtx.range_pop()

      end1.record(stream_dec)
      end2.record(stream_prefill)

      stream_dec.synchronize()
      stream_prefill.synchronize()


      elapsedTime1 = start1.elapsed_time(end1)
      elapsedTime2 = start2.elapsed_time(end2)

      if elapsedTime2 > 1.2 * elapsedTime1:
        #green context running prefill should never be idle for this benchmark to work
        break

      prefill_iters *= 2


    if (elapsedTime2 <= 1.2*elapsedTime1): # prefill iter scaling got capped; abort run
      print(f"WARN: prefill didn't cover decode at active_sms={active_sms}, skipping")
      continue

    itl = elapsedTime1 / NUM_ITERS

    print(f"(Batch Size: {B_dec}, Active SMs: {active_sms}) inter_token_latency (ms): {itl:.3f} Decode total time (ms): {elapsedTime1:.1f} Prefill total time (ms): {elapsedTime2:.1f}")

    contention_decode_results.append({"Batch": B_dec, "Active_SMs": active_sms, "ITL_ms": round(itl, 3)})

  return contention_decode_results


