import argparse
import torch
import flashinfer
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
import model

partitions = {}
def get_green_ctx(active_sms):
  if active_sms not in partitions:
    dev = torch.device("cuda:0")
    partitions[active_sms] = split_device_green_ctx_by_sm_count(dev, [active_sms])

  return partitions[active_sms]

def measure_decode_prefill_contention(mdl, B_dec, S_prefill, S_dec, decode_sms):

  # Experiment: Profile decodes with serial prefill running in the other green context.
  # Since we are wanting to collect HW counters info (using nsight compute) and these can't
  # be attributed to either prefill or decode, we have to ensure that what we're measuring is
  # really under contention: launching enough prefills to cover decodes in not sufficient, their
  # timeline must approx match

  # Set up KV caches on GPU HBM
  paged_kv = torch.randn((B_dec, 2, S_dec, mdl.N, mdl.H), dtype = torch.float16, device = "cuda")
 
  activation_decode = torch.randn((B_dec, 1, mdl.D), dtype = torch.float16, device = "cuda")

  activation_prefill = torch.randn((1, S_prefill, mdl.D), dtype = torch.float16, device = "cuda") 

  # Set up FlashInfer wrapper object for decode

  # Set up paging config of KV cache
  workspace_dec = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_dec, "NHD")

  kv_indptr = torch.arange(0, B_dec + 1, dtype = torch.int32, device = "cuda")
  kv_indices = torch.arange(0, B_dec, dtype = torch.int32, device = "cuda")
  kv_last_page_len = torch.full((B_dec, ), S_dec, dtype = torch.int32, device = "cuda")

  decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads = mdl.N, num_kv_heads = mdl.N, head_dim = mdl.H, page_size = S_dec)

  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  streams, resources = get_green_ctx(decode_sms) 
  stream_dec = streams[0] 
  stream_prefill = streams[1] 

  with torch.cuda.stream(stream_dec):
      # Warmup: launch a decode kernel
      with torch.inference_mode():
        _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)

  with torch.cuda.stream(stream_prefill):
      # Warmup: launch a prefill kernel
      with torch.inference_mode():
        _ = mdl.do_serial_prefill(activation_prefill)

  torch.cuda.synchronize()

  prefill_iters = 1
  decode_iters = 2
  covered = False

  for _ in range(20):
    start1 = torch.cuda.Event(enable_timing = True)
    end1 = torch.cuda.Event(enable_timing = True)
    start2 = torch.cuda.Event(enable_timing = True)
    end2 = torch.cuda.Event(enable_timing = True)

    torch.cuda.synchronize()

    start1.record(stream_prefill)

    with torch.cuda.stream(stream_prefill):
      with torch.inference_mode():
        for _ in range(prefill_iters):
            _ = mdl.do_serial_prefill(activation_prefill)

    start2.record(stream_dec)

    with torch.cuda.stream(stream_dec):
      with torch.inference_mode():
        for _ in range(decode_iters):
          out = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)

    end1.record(stream_prefill)
    end2.record(stream_dec)

    stream_dec.synchronize()
    stream_prefill.synchronize()


    elapsedTime1 = start1.elapsed_time(end1)
    elapsedTime2 = start2.elapsed_time(end2)

    if elapsedTime2 > 1.05 * elapsedTime1:
      covered = True
      break

    decode_iters *= 2

  assert(covered is True)

  torch.cuda.synchronize()

  torch.cuda.nvtx.range_push("prefill")
  with torch.cuda.stream(stream_prefill):
      with torch.inference_mode():
        for _ in range(prefill_iters):
          _ = mdl.do_serial_prefill(activation_prefill)
  torch.cuda.nvtx.range_pop()

  torch.cuda.synchronize()

  torch.cuda.nvtx.range_push("decode")
  with torch.cuda.stream(stream_dec):
      with torch.inference_mode():
        for _ in range(decode_iters):
          _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
  torch.cuda.nvtx.range_pop()

  torch.cuda.synchronize()

  torch.cuda.nvtx.range_push("prefill-decode")
  with torch.cuda.stream(stream_prefill):
      with torch.inference_mode():
          for _ in range(prefill_iters):
            _ = mdl.do_serial_prefill(activation_prefill)

  with torch.cuda.stream(stream_dec):
      with torch.inference_mode():
          for _ in range(decode_iters):
            _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
  torch.cuda.nvtx.range_pop()

  torch.cuda.synchronize()

def main():
  p = argparse.ArgumentParser()
  p.add_argument("-D", "--dim", type=int, required=True, help="model hidden dim D")
  p.add_argument("-N", "--heads", type=int, required=True, help="num attention heads N")
  p.add_argument("--decode-batch", type=int, required=True, help="decode batch size B_dec")
  p.add_argument("-S-prefill", "--seq-len-prefill", type=int, required=True, help="prefill sequence length")
  p.add_argument("-S-decode", "--seq-len-decode", type=int, required=True, help="decode sequence length")
  p.add_argument("--decode-sms", type=int, required=True, help="Number of SMs in decode green context")
  args = p.parse_args()

  print(f"Profiling prefill decode contention under green context: "
        f"D={args.dim} N={args.heads} "
        f"B_dec={args.decode_batch} S_prefill={args.seq_len_prefill} S_dec={args.seq_len_decode} "
        f"SMs={args.decode_sms}")

  mdl = model.Model(args.dim, args.heads)
  measure_decode_prefill_contention(mdl, args.decode_batch, args.seq_len_prefill, args.seq_len_decode, args.decode_sms)

  print("Completed profiling")

if __name__ == "__main__":
  main()
