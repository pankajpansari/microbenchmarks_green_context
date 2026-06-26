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

def balance_iters(mdl, B_dec, S_prefill, S_dec, decode_sms):
  paged_kv = torch.randn((B_dec, 2, S_dec, mdl.N, mdl.H), dtype=torch.float16, device="cuda")
  activation_decode = torch.randn((B_dec, 1, mdl.D), dtype=torch.float16, device="cuda")
  activation_prefill = torch.randn((1, S_prefill, mdl.D), dtype=torch.float16, device="cuda")

  workspace_dec = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
  decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_dec, "NHD")

  kv_indptr = torch.arange(0, B_dec + 1, dtype=torch.int32, device="cuda")
  kv_indices = torch.arange(0, B_dec, dtype=torch.int32, device="cuda")
  kv_last_page_len = torch.full((B_dec,), S_dec, dtype=torch.int32, device="cuda")

  decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads=mdl.N, num_kv_heads=mdl.N, head_dim=mdl.H, page_size=S_dec)

  streams, resources = get_green_ctx(decode_sms)
  stream_dec = streams[0]
  stream_prefill = streams[1]

  prefill_iters = 4 
  decode_iters = 20

  with torch.cuda.stream(stream_prefill):
    with torch.inference_mode():
      for _ in range(5):
        _ = mdl.do_serial_prefill(activation_prefill)
  torch.cuda.synchronize()

  start1 = torch.cuda.Event(enable_timing=True)
  end1 = torch.cuda.Event(enable_timing=True)
  with torch.cuda.stream(stream_prefill):
    start1.record()
    with torch.inference_mode():
      for _ in range(prefill_iters):
        _ = mdl.do_serial_prefill(activation_prefill)
    end1.record()
  torch.cuda.synchronize()
  elapsed_prefill = start1.elapsed_time(end1)

  with torch.cuda.stream(stream_dec):
    with torch.inference_mode():
      for _ in range(5):
        _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
  torch.cuda.synchronize()

  start2 = torch.cuda.Event(enable_timing=True)
  end2 = torch.cuda.Event(enable_timing=True)
  with torch.cuda.stream(stream_dec):
    start2.record()
    with torch.inference_mode():
      for _ in range(decode_iters):
        _ = mdl.do_batched_decode(activation_decode, paged_kv, decode_wrapper)
    end2.record()
  torch.cuda.synchronize()
  elapsed_decode = start2.elapsed_time(end2)

  if elapsed_prefill > elapsed_decode:
    decode_iters = int(decode_iters * (elapsed_prefill / elapsed_decode))
  else:
    prefill_iters = int(prefill_iters * (elapsed_decode / elapsed_prefill))

  print(f"prefill_iters={prefill_iters}")
  print(f"decode_iters={decode_iters}")

def main():
  p = argparse.ArgumentParser()
  p.add_argument("-D", "--dim", type=int, required=True)
  p.add_argument("-N", "--heads", type=int, required=True)
  p.add_argument("--decode-batch", type=int, required=True)
  p.add_argument("-S-prefill", "--seq-len-prefill", type=int, required=True)
  p.add_argument("-S-decode", "--seq-len-decode", type=int, required=True)
  p.add_argument("--decode-sms", type=int, required=True)
  args = p.parse_args()

  mdl = model.Model(args.dim, args.heads)
  balance_iters(mdl, args.decode_batch, args.seq_len_prefill, args.seq_len_decode, args.decode_sms)

if __name__ == "__main__":
  main()
