import torch
import torch.nn.functional as Func
import flashinfer
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
import pandas as pd
import einops
import subprocess, json, sys, socket, datetime
import os

_ = torch.empty(1, device='cuda:0')

S_dec= 2048 # sequence len
D = 8192 # model embedding dim
N = 32 # number of heads (same for K,Q,V)
H = D // N # head dimension
F = 4 * D # FFN hidden dimension
B_dec_sweep =[32, 64, 128, 256, 512]
S_prefill_sweep = [512, 1024, 2048]

W_k = torch.randn((D, N, H), dtype = torch.float16, device = "cuda")
W_q = torch.randn((D, N, H), dtype = torch.float16, device = "cuda")
W_v = torch.randn((D, N, H), dtype = torch.float16, device = "cuda")
W_o = torch.randn((N, H, D), dtype = torch.float16, device = "cuda")

W_in = torch.randn((D, F), dtype = torch.float16, device = "cuda")
W_out = torch.randn((F, D), dtype = torch.float16, device = "cuda")

def do_prefill(x):
  
  # B = 1; single request can saturate compute
  # self-attention block
  q = torch.einsum('sd, dnh -> snh', x, W_q)   

  k = torch.einsum('sd, dnh -> snh', x, W_k)   
  v = torch.einsum('sd, dnh -> snh', x, W_v)   
  
  v_attention = flashinfer.single_prefill_with_kv_cache(q, k, v, causal = True)

  o = torch.einsum('snh, nhd -> sd', v_attention, W_o)

  # ffn block
  o_1 = torch.einsum('sd, df -> sf', o, W_in)
  o_1 = Func.relu(o_1)
  output = torch.einsum('sf, fd -> sd', o_1, W_out)
  return output

def do_serial_prefill(x):
  
  # Iterate over samples in batch
  # self-attention block
  output = torch.zeros_like(x)

  for i in range(x.shape[0]):
    q_i = torch.einsum('sd, dnh -> snh', x[i], W_q)   

    k_i = torch.einsum('sd, dnh -> snh', x[i], W_k)   
    v_i = torch.einsum('sd, dnh -> snh', x[i], W_v)   
    
    v_attention_i = flashinfer.single_prefill_with_kv_cache(q_i, k_i, v_i, causal = True)

    o_i = torch.einsum('snh, nhd -> sd', v_attention_i, W_o)

    # ffn block
    o_1_i = torch.einsum('sd, df -> sf', o_i, W_in)
    o_1_i = Func.relu(o_1_i)
    output[i] = torch.einsum('sf, fd -> sd', o_1_i, W_out)
  return output

def do_batched_prefill(x, prefill_wrapper):
  
  #x is a batch of prefill token embedding 
  b, s = x.shape[0], x.shape[1]
  q = torch.einsum('bsd, dnh -> bsnh', x, W_q).reshape(b * s, N, H)

  #we don't append these to kv cache
  k = torch.einsum('bsd, dnh -> bsnh', x, W_k).reshape(b * s, N, H)
  v = torch.einsum('bsd, dnh -> bsnh', x, W_v).reshape(b * s, N, H)

  v_attention = prefill_wrapper.run(q, k, v).reshape(b, s, N, H)

  o = torch.einsum('bsnh, nhd -> bsd', v_attention, W_o) 

  #ffn block
  o_1 = torch.einsum('bsd, df -> bsf', o, W_in) 
  o_1 = Func.relu(o_1)
  output = torch.einsum('bsf, fd -> bsd', o_1, W_out) 
  return output


def do_batched_decode(x, paged_kv, decode_wrapper):
  
  # context len always at S_dec; cache does not grow

  #x is a batch of last generate token embedding 
  q = torch.einsum('bsd, dnh -> bsnh', x, W_q).squeeze(1)   # s = 1 

  #we don't append these to kv cache
  k = torch.einsum('bsd, dnh -> bsnh', x, W_k).squeeze(1)   # s = 1 
  v = torch.einsum('bsd, dnh -> bsnh', x, W_v).squeeze(1)   # s = 1 

  v_attention = decode_wrapper.run(q, paged_kv).unsqueeze(1)

  o = torch.einsum('bsnh, nhd -> bsd', v_attention, W_o) # s = 1 

  #ffn block
  o_1 = torch.einsum('bsd, df -> bsf', o, W_in) # s = 1 
  o_1 = Func.relu(o_1)
  output = torch.einsum('bsf, fd -> bsd', o_1, W_out) # s = 1 
  return output

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
  all_streams = []
  all_resources = []
  granularity = 8

  NUM_WARMUPS = 10
  NUM_ITERS= 100 
  no_contention_results = []

  # Without any partition - decodes have all SMs
  # Warmup: launch some decode kernel
  with torch.inference_mode():
    for _ in range(NUM_WARMUPS):
      out = do_batched_decode(activation, paged_kv, decode_wrapper)

  start = torch.cuda.Event(enable_timing = True)
  end = torch.cuda.Event(enable_timing = True)

  torch.cuda.synchronize()

  start.record()

  # Actual decode runs for timing
  with torch.inference_mode():
    for _ in range(NUM_ITERS):
      out = do_batched_decode(activation, paged_kv, decode_wrapper)

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
          out = do_batched_decode(activation, paged_kv, decode_wrapper)

      start = torch.cuda.Event(enable_timing = True)
      end = torch.cuda.Event(enable_timing = True)

      target_stream.synchronize()

      start.record()

      # Actual decode runs for timing
      with torch.inference_mode():
        for _ in range(NUM_ITERS):
          out = do_batched_decode(activation, paged_kv, decode_wrapper)

      end.record()

      target_stream.synchronize()

      itl = start.elapsed_time(end) / NUM_ITERS

      print(f"Without contention (Batch Size: {B}, Active SMs: {active_sms}) inter_token_latency (ms): {itl:.3f}")

      no_contention_results.append({"Batch": B, "Active_SMs": active_sms, "Elapsed_time_ms": round(itl, 3)})

  return no_contention_results

def measure_decode_under_prefill_contention(prefill_fn, B_dec, B_prefill, S_prefill):
  # Experiment: Profile decodes with serial prefill running in the other green context 

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
          out = do_batched_decode(activation_decode, paged_kv, decode_wrapper)

    with torch.cuda.stream(stream_prefill):
      # Warmup: launch some prefill kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS / 2):
          if prefill_fn is do_batched_prefill:
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
            if prefill_fn is do_batched_prefill:
              out = prefill_fn(activation_prefill, prefill_wrapper)
            else:
              out = prefill_fn(activation_prefill)
      torch.cuda.nvtx.range_pop()

      start1.record(stream_dec)

      torch.cuda.nvtx.range_push("decode")
      with torch.cuda.stream(stream_dec):
        with torch.inference_mode():
          for _ in range(NUM_ITERS):
            out = do_batched_decode(activation_decode, paged_kv, decode_wrapper)
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

def contention_greenctx_prefill(prefill_fn, B_dec, B_prefill, S_prefill):
  # Experiment: Profile prefills with decodes running in the other green context 

  # Set up KV caches on GPU HBM
  paged_kv = torch.randn((B_dec, 2, S_dec, N, H), dtype = torch.float16, device = "cuda")
 
  activation_decode = torch.randn((B_dec, 1, D), dtype = torch.float16, device = "cuda")

  activation_prefill = torch.randn((B_prefill, S_prefill, D), dtype = torch.float16, device = "cuda") 

  # Set up FlashInfer wrapper object for decode

  # Set up paging config of KV cache
  workspace_dec = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_dec, "NHD")

  page_size = S_dec
  num_pages = B_dec
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
  all_streams = []
  all_resources = []
  granularity = 8

  NUM_WARMUPS = 5
  NUM_ITERS = 10
  contention_prefill_results = []

  # Sweep over partition configs. Create green context + associated stream for each 

  for i in range(1, num_sms // granularity):

    active_sms = num_sms - i*granularity
    # Create green contexts
    streams, resources = split_device_green_ctx_by_sm_count(dev, [active_sms])

    stream_prefill = streams[0] 
    stream_dec = streams[1] 

    with torch.cuda.stream(stream_dec):
      # Warmup: launch some decode kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS):
          out = do_batched_decode(activation_decode, paged_kv, decode_wrapper)

    with torch.cuda.stream(stream_prefill):
      # Warmup: launch some prefill kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS):
          if prefill_fn is do_batched_prefill:
            out = prefill_fn(activation_prefill, prefill_wrapper)
          else:
            out = prefill_fn(activation_prefill)

    decode_iters = 100*NUM_ITERS

    for _ in range(6):
      start1 = torch.cuda.Event(enable_timing = True)
      end1 = torch.cuda.Event(enable_timing = True)
      start2 = torch.cuda.Event(enable_timing = True)
      end2 = torch.cuda.Event(enable_timing = True)

      torch.cuda.synchronize()

      start1.record(stream_dec)

      torch.cuda.nvtx.range_push("decode")
      # ensure decodes are underway before prefills start executing from other stream
      with torch.cuda.stream(stream_dec):
        with torch.inference_mode():
          for _ in range(decode_iters):
            out = do_batched_decode(activation_decode, paged_kv, decode_wrapper)
      torch.cuda.nvtx.range_pop()

      start2.record(stream_prefill)

      torch.cuda.nvtx.range_push("prefill")
      with torch.cuda.stream(stream_prefill):
        with torch.inference_mode():
          for _ in range(NUM_ITERS):
            if prefill_fn is do_batched_prefill:
              out = prefill_fn(activation_prefill, prefill_wrapper)
            else:
              out = prefill_fn(activation_prefill)
      torch.cuda.nvtx.range_pop()

      end1.record(stream_dec)
      end2.record(stream_prefill)

      stream_dec.synchronize()
      stream_prefill.synchronize()

      elapsedTime1 = start1.elapsed_time(end1)
      elapsedTime2 = start2.elapsed_time(end2)

      if elapsedTime1 > 1.2 * elapsedTime2:
      #green context running decode should never be idle for this benchmark to work
        break

      decode_iters *= 2

    if (elapsedTime1 <= 1.2*elapsedTime2): # decode iter scaling got capped; abort run
      print(f"WARN: decode didn't cover prefill at active_sms={active_sms}, skipping")
      continue

    prefill_tp = int((B_prefill * S_prefill * NUM_ITERS * 1000)/ elapsedTime2) # Num of tokens processed/time (toks/s)

    print(f"(Decode Batch Size: {B_dec}, Active SMs: {active_sms}) througput (toks/s): {prefill_tp}")

    contention_prefill_results.append({"Active_SMs": active_sms, "Decode_batch": B_dec, "Prefill_contx_len": S_prefill, "Throughput_toks_s": prefill_tp})

  return contention_prefill_results


def no_contention_greenctx_prefill(B_prefill):
  # Setting: Prefill using one green context; remaining SMs outside context idle 

  activation = torch.randn((B_prefill, S, D), dtype = torch.float16, device = "cuda")

  # Set up FlashInfer wrapper object for prefill
  workspace_prefill = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  qo_indptr = torch.arange(0, (B_prefill + 1) * S, S, dtype=torch.int32, device="cuda:0")
  kv_indptr = qo_indptr.clone()
  prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_prefill, "NHD")
  prefill_wrapper.plan(qo_indptr, kv_indptr, num_qo_heads = N, num_kv_heads = N, head_dim_qk = H, causal=True)

  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  dev = torch.device("cuda:0")
  all_streams = []
  all_resources = []
  granularity = 8

  NUM_WARMUPS_dec= 10
  NUM_ITERS_dec= 100 
  no_contention_results = []

  # Without any partition - prefill has all SMs
  # Warmup: launch some prefill kernels
  with torch.inference_mode():
    for _ in range(NUM_WARMUPS):
      out = do_batched_prefill(activation, prefill_wrapper)

  start = torch.cuda.Event(enable_timing = True)
  end = torch.cuda.Event(enable_timing = True)

  torch.cuda.synchronize()

  start.record()

  # Actual prefill runs for timing
  with torch.inference_mode():
    for _ in range(NUM_ITERS):
      out = do_batched_prefill(activation, prefill_wrapper)

  end.record()

  torch.cuda.synchronize()

  prefill_tp = int((B_prefill * S_dec* NUM_ITERS_dec* 1000)/ start.elapsed_time(end)) # Num of tokens processed/time (toks/s)

  print(f"Without contention (Active SMs: {num_sms}) througput (toks/s): {prefill_tp}")

  no_contention_results.append({"Active_SMs": num_sms, "Throughput_toks_s": prefill_tp})

  # Sweep over partition configs. Create green context + associated stream for each 
  for i in range(1, num_sms // granularity):

    active_sms = num_sms - i*granularity
    streams, resources = split_device_green_ctx_by_sm_count(dev, [active_sms])
    target_stream = streams[0]
    with torch.cuda.stream(target_stream):

      # Warmup: launch some prefill kernel
      with torch.inference_mode():
        for _ in range(NUM_WARMUPS):
          out = do_batched_prefill(activation, prefill_wrapper)

      start = torch.cuda.Event(enable_timing = True)
      end = torch.cuda.Event(enable_timing = True)

      target_stream.synchronize()

      start.record()

      # Actual prefill runs for timing
      with torch.inference_mode():
        for _ in range(NUM_ITERS):
          out = do_batched_prefill(activation, prefill_wrapper)

      end.record()

      target_stream.synchronize()

      prefill_tp = int((B_prefill * S_dec* NUM_ITERS_dec* 1000)/ start.elapsed_time(end)) # Num of tokens processed/time (toks/s)

      print(f"Without contention (Active SMs: {active_sms}) througput (toks/s): {prefill_tp}")

      no_contention_results.append({"Active_SMs": active_sms, "Throughput_toks_s": prefill_tp})

  return no_contention_results

def generate_csv_header_sidecar(df, params, out_dir = "data"):
  
  commit_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
  dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, 
                         text=True).stdout.strip())
  gpu = torch.cuda.get_device_name(0)
  now = datetime.datetime.now()
  timestamp = now.isoformat(timespec='seconds')
  command = " ".join(sys.argv)

  stamp = now.strftime("%Y%m%d-%H%M")
  gpu_slug = torch.cuda.get_device_name(0).replace(" ", "-")
  csv_filename = f"{out_dir}/{params['exp']}__d{params['D']}__{gpu_slug}__{stamp}.csv"

  meta = {"commit_hash": commit_hash, "dirty": dirty, "gpu": gpu, "timestamp": timestamp, 
          "command": command, "params": params}
  
  os.makedirs(out_dir, exist_ok = True)
  with open(f"{csv_filename}.meta.json", "w") as f:
    json.dump(meta, f, indent = 2)  

  header = "# " + " ".join(f"{k}={meta[k]}" for k in ["commit_hash", "dirty", "gpu", "timestamp"]) 

  with open(csv_filename, "w", newline="") as f:
    f.write(header + "\n")
    df.to_csv(f, index=False)


def run_decode_isolated_experiment():

  all_batch_results = []
  for B in B_dec_sweep: # batch size
    no_contention_results = measure_decode_isolated(B)
    all_batch_results.append(no_contention_results)

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])
  
  params = {"B_dec_sweep": B_dec_sweep, "S_dec": S_dec, "D": D, "num_kv_heads": N, "H": H, 
            "F": F, "exp": "decode_only_no_contention"}

  generate_csv_header_sidecar(df, params)

def run_decode_vs_prefill_experiment(prefill_fn):

  B_prefill = 16 
  all_batch_results = []

  for S_prefill in S_prefill_sweep:
    for B_dec in B_dec_sweep: # batch size
      contention_decode_results = measure_decode_under_prefill_contention(prefill_fn, B_dec, B_prefill, S_prefill)
      all_batch_results.append(contention_decode_results)

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])

  params = {"B_dec_sweep": B_dec_sweep, "S_dec": S_dec, "S_prefill_sweep": S_prefill_sweep, "D": D, 
            "num_kv_heads": N, "H": H, "F": F, "prefill_fn": prefill_fn.__name__, 
            "exp": f"decode_with_prefill_contention_{prefill_fn.__name__}"}

  generate_csv_header_sidecar(df, params)

def run_prefill_only_exp():

  B_prefill = 8
  no_contention_results = no_contention_greenctx_prefill(B_prefill)

  df = pd.DataFrame(no_contention_results)
  csv_filename = "greenctx_no_contention_batched_prefill_tp" + "_d" + str(D) + ".csv"
  df.to_csv(csv_filename, index = False)

def run_prefill_contention_exp():

  B_prefill = 8 
  S_prefill = 2048
  all_batch_results = []
  for B_dec in B_dec_sweep: # batch size
    contention_prefill_results = contention_greenctx_prefill(do_serial_prefill, B_dec, B_prefill, S_prefill)
    all_batch_results.append(contention_prefill_results)

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])

  params = {"B_dec_sweep": B_dec_sweep, "S_dec": S_dec, "S_prefill": S_prefill, "D": D, 
            "num_kv_heads": N, "H": H, "F": F, "exp": "prefill_with_decode_contention"}

  generate_csv_header_sidecar(df, params)


def main():
  # Decode with no prefill contention; varying batch size, partition configs 
  run_decode_isolated_experiment()

  # Decode contention with serial prefill in other green context; varying prefill context length
  run_decode_vs_prefill_experiment(do_serial_prefill)

  # Decode contention with batched prefill in other green context; fixed prefill context length 
  run_decode_vs_prefill_experiment(do_batched_prefill)

#  run_prefill_contention_exp()
#  run_prefill_only_exp()

if __name__ == "__main__":
  main()
