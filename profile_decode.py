import argparse
import torch
import flashinfer
import model

def measure_decode_all_sms(mdl, B, S_dec):
  # Setting: Decodes using all SMs 

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

  NUM_WARMUPS = 1 
  NUM_ITERS= 1 

  # Without any partition - decodes have all SMs
  # Warmup: launch some decode kernel
  with torch.inference_mode():
    for _ in range(NUM_WARMUPS):
      _ = mdl.do_batched_decode(activation, paged_kv, decode_wrapper)

  torch.cuda.synchronize()

  # Actual decode runs for timing
  torch.cuda.nvtx.range_push("decode-all-sms")
  with torch.inference_mode():
    _ = mdl.do_batched_decode(activation, paged_kv, decode_wrapper)
  torch.cuda.nvtx.range_pop()

  torch.cuda.synchronize()

def main():
  p = argparse.ArgumentParser()
  p.add_argument("-D", "--dim", type=int, required=True, help="model hidden dim D")
  p.add_argument("-N", "--heads", type=int, required=True, help="num attention heads N")
  p.add_argument("--batch", type=int, required=True, help="decode batch size B")
  p.add_argument("-S", "--seq-len", type=int, required=True, help="decode sequence length")
  args = p.parse_args()

  mdl = model.Model(args.dim, args.heads)
  measure_decode_all_sms(mdl, args.batch, args.seq_len)

  print("Completed profiling")

if __name__ == "__main__":
  main()
