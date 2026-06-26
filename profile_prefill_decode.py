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

def measure_decode_prefill_contention(mdl, B_dec, S_prefill, S_dec, decode_sms, prefill_iters, decode_iters):

  paged_kv = torch.randn((B_dec, 2, S_dec, mdl.N, mdl.H), dtype = torch.float16, device = "cuda")
  activation_decode = torch.randn((B_dec, 1, mdl.D), dtype = torch.float16, device = "cuda")
  activation_prefill = torch.randn((1, S_prefill, mdl.D), dtype = torch.float16, device = "cuda")

  workspace_dec = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_dec, "NHD")

  kv_indptr = torch.arange(0, B_dec + 1, dtype = torch.int32, device = "cuda")
  kv_indices = torch.arange(0, B_dec, dtype = torch.int32, device = "cuda")
  kv_last_page_len = torch.full((B_dec, ), S_dec, dtype = torch.int32, device = "cuda")

  decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads = mdl.N, num_kv_heads = mdl.N, head_dim = mdl.H, page_size = S_dec)

  streams, resources = get_green_ctx(decode_sms)
  stream_dec = streams[0]
  stream_prefill = streams[1]

  with torch.cuda.stream(stream_dec):
    with torch.inference_mode():
      _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)

  with torch.cuda.stream(stream_prefill):
    with torch.inference_mode():
      _ = mdl.do_serial_prefill(activation_prefill)

  torch.cuda.synchronize()

  torch.cuda.nvtx.range_push("prefill")
  with torch.cuda.stream(stream_prefill):
    with torch.inference_mode():
      for _ in range(prefill_iters):
        _ = mdl.do_serial_prefill(activation_prefill)
  torch.cuda.synchronize()
  torch.cuda.nvtx.range_pop()

  torch.cuda.nvtx.range_push("decode")
  with torch.cuda.stream(stream_dec):
    with torch.inference_mode():
      for _ in range(decode_iters):
        _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
  torch.cuda.synchronize()
  torch.cuda.nvtx.range_pop()

  start1 = torch.cuda.Event(enable_timing = True)
  end1 = torch.cuda.Event(enable_timing = True)
  start2 = torch.cuda.Event(enable_timing = True)
  end2 = torch.cuda.Event(enable_timing = True)

  torch.cuda.nvtx.range_push("prefill-decode")
  gate = torch.cuda.Event(enable_timing=False)
  stream_prefill.wait_event(gate)
  stream_dec.wait_event(gate)

  with torch.cuda.stream(stream_prefill):
    start1.record()
    with torch.inference_mode():
      for _ in range(prefill_iters):
        _ = mdl.do_serial_prefill(activation_prefill)
    end1.record()
  with torch.cuda.stream(stream_dec):
    start2.record()
    with torch.inference_mode():
      for _ in range(decode_iters):
        _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
    end2.record()
  gate.record()  # fires immediately (default stream has no pending work), releases both streams
  torch.cuda.synchronize()
  torch.cuda.nvtx.range_pop()

  elapsed_prefill = start1.elapsed_time(end1)
  elapsed_decode = start2.elapsed_time(end2)
  print(f"prefill_duration={elapsed_prefill} decode_duration={elapsed_decode}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("-D", "--dim", type=int, required=True, help="model hidden dim D")
  p.add_argument("-N", "--heads", type=int, required=True, help="num attention heads N")
  p.add_argument("--decode-batch", type=int, required=True, help="decode batch size B_dec")
  p.add_argument("-S-prefill", "--seq-len-prefill", type=int, required=True, help="prefill sequence length")
  p.add_argument("-S-decode", "--seq-len-decode", type=int, required=True, help="decode sequence length")
  p.add_argument("--decode-sms", type=int, required=True, help="Number of SMs in decode green context")
  p.add_argument("--prefill-iters", type=int, required=True, help="number of prefill iterations")
  p.add_argument("--decode-iters", type=int, required=True, help="number of decode iterations")
  args = p.parse_args()

  print(f"Profiling prefill decode contention under green context: "
        f"D={args.dim} N={args.heads} "
        f"B_dec={args.decode_batch} S_prefill={args.seq_len_prefill} S_dec={args.seq_len_decode} "
        f"SMs={args.decode_sms} prefill_iters={args.prefill_iters} decode_iters={args.decode_iters}")

  mdl = model.Model(args.dim, args.heads)
  measure_decode_prefill_contention(
    mdl, args.decode_batch, args.seq_len_prefill, args.seq_len_decode,
    args.decode_sms, args.prefill_iters, args.decode_iters)

  print("Completed profiling")

if __name__ == "__main__":
  main()
