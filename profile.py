import torch
import flashinfer
from config import S_dec 
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
import model
import sys

partitions = {}
def get_green_ctx(active_sms):
  if active_sms not in partitions:
    dev = torch.device("cuda:0")
    partitions[active_sms] = split_device_green_ctx_by_sm_count(dev, [active_sms])

  return partitions[active_sms]

def measure_decode_isolated(mdl, B):
  # Setting: Decodes using one green context; remaining SMs outside context idle 

  # Set up KV caches on GPU HBM
  k_cache = torch.randn((B, S_dec, mdl.N, mdl.H), dtype = torch.float16, device = "cuda")
  v_cache = torch.randn((B, S_dec, mdl.N, mdl.H), dtype = torch.float16, device = "cuda")
  paged_kv = torch.stack([k_cache, v_cache], dim = 1)
 
  activation = torch.randn((B, 1, mdl.D), dtype = torch.float16, device = "cuda")

  # Set up paging config of KV cache
  workspace = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

  kv_indptr = torch.arange(0, B + 1, dtype = torch.int32, device = "cuda")
  kv_indices = torch.arange(0, B, dtype = torch.int32, device = "cuda")
  kv_last_page_len = torch.full((B, ), S_dec, dtype = torch.int32, device = "cuda")

  # Set up FlashInfer wrapper object for decode

  decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads = mdl.N, num_kv_heads = mdl.N, head_dim = mdl.H, page_size = S_dec)


  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  dev = torch.device("cuda:0")
  granularity = 8

  NUM_WARMUPS = 1 
  NUM_ITERS= 2 
  no_contention_results = []

  # Without any partition - decodes have all SMs
  # Warmup: launch some decode kernel
  with torch.inference_mode():
    for _ in range(NUM_WARMUPS):
      _ = mdl.do_batched_decode(activation, paged_kv, decode_wrapper)

  start = torch.cuda.Event(enable_timing = True)
  end = torch.cuda.Event(enable_timing = True)

  torch.cuda.synchronize()

  start.record()

  # Actual decode runs for timing
  with torch.inference_mode():
    for _ in range(NUM_ITERS):
      _ = mdl.do_batched_decode(activation, paged_kv, decode_wrapper)

  end.record()

  torch.cuda.synchronize()

  itl = start.elapsed_time(end) / NUM_ITERS

  no_contention_results.append({"Batch": B, "Active_SMs": num_sms, "ITL_ms": round(itl, 3)})


  # Sweep over partition configs. Create green context + associated stream for each 

  active_sms = 72 
  streams, resources = get_green_ctx(active_sms) 

  target_stream = streams[0]
  with torch.cuda.stream(target_stream):

    # Warmup: launch some decode kernel
    with torch.inference_mode():
      for _ in range(NUM_WARMUPS):
        out = mdl.do_batched_decode(activation, paged_kv, decode_wrapper)

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)

    target_stream.synchronize()

    start.record()

    torch.cuda.nvtx.range_push("decode-only")
    # Actual decode runs for timing
    with torch.inference_mode():
      for _ in range(NUM_ITERS):
        out = mdl.do_batched_decode(activation, paged_kv, decode_wrapper)
    torch.cuda.nvtx.range_pop()

    end.record()

    target_stream.synchronize()

def main():
  print(f"Profiling decode only in green context")
  mdl = model.Model(sys.argv[1], sys.argv[2])

  # Decode with no prefill contention; varying batch size, partition configs 
  measure_decode_isolated(mdl, sys.argv[3])
  print(f"Completed profiling")

if __name__ == "__main__":
  main()