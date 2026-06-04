import torch
import torch.nn.functional as Func
import flashinfer
from torch.nn.attention import SDPBackend, sdpa_kernel
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
import pandas as pd

_ = torch.empty(1, device='cuda:0')

S = 2048 # sequence len
D = 8192 # model embedding dim
N = 32 # number of heads (same for K,Q,V)
H = D // N # head dimension
F = 4 * D # FFN hidden dimension

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

def do_batched_decode(x, paged_kv, decode_wrapper):
  
  # context len always at S; cache does not grow

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

def no_contention_greenctx_decodes(decode_wrapper, B):
  # Setting: Decodes using one green context; remaining SMs outside context idle 

  # Set up KV caches on GPU HBM
  k_cache = torch.randn((B, S, N, H), dtype = torch.float16, device = "cuda")
  v_cache = torch.randn((B, S, N, H), dtype = torch.float16, device = "cuda")
  paged_kv = torch.stack([k_cache, v_cache], dim = 1)
 
  activation = torch.randn((B, 1, D), dtype = torch.float16, device = "cuda")

  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  dev = torch.device("cuda:0")
  all_streams = []
  all_resources = []
  granularity = 8

  NUM_WARMUPS = 10
  NUM_ITERS = 100 
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
  print(f"B = {B} no contention green context benchmark done")


def no_contention_greenctx_prefill():
  # Setting: Prefill using one green context; remaining SMs outside context idle 

  activation = torch.randn((S, D), dtype = torch.float16, device = "cuda")

  device_props = torch.cuda.get_device_properties(0)
  num_sms = device_props.multi_processor_count

  dev = torch.device("cuda:0")
  all_streams = []
  all_resources = []
  granularity = 8

  NUM_WARMUPS = 10
  NUM_ITERS = 100 
  no_contention_results = []

  # Without any partition - prefill has all SMs
  # Warmup: launch some prefill kernels
  with torch.inference_mode():
    for _ in range(NUM_WARMUPS):
      out = do_prefill(activation)

  start = torch.cuda.Event(enable_timing = True)
  end = torch.cuda.Event(enable_timing = True)

  torch.cuda.synchronize()

  start.record()

  # Actual prefill runs for timing
  with torch.inference_mode():
    for _ in range(NUM_ITERS):
      out = do_prefill(activation)

  end.record()

  torch.cuda.synchronize()

  prefill_tp = int((S * NUM_ITERS * 1000)/ start.elapsed_time(end)) # Num of tokens processed/time (toks/s)

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
          out = do_prefill(activation)

      start = torch.cuda.Event(enable_timing = True)
      end = torch.cuda.Event(enable_timing = True)

      target_stream.synchronize()

      start.record()

      # Actual prefill runs for timing
      with torch.inference_mode():
        for _ in range(NUM_ITERS):
          out = do_prefill(activation)

      end.record()

      target_stream.synchronize()

      prefill_tp = int((S * NUM_ITERS * 1000)/ start.elapsed_time(end)) # Num of tokens processed/time (toks/s)

      print(f"Without contention (Active SMs: {active_sms}) througput (toks/s): {prefill_tp}")

      no_contention_results.append({"Active_SMs": active_sms, "Throughput_toks_s": prefill_tp})

  return no_contention_results

def run_decode_only_exp():
  # Set up paging config of KV cache
  workspace = torch.zeros(128 * 1024 * 1024, dtype = torch.uint8, device = "cuda")

  all_batch_results = []
  for B in [32, 64, 128, 256, 512]: # batch size
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    page_size = S
    num_pages = B
    kv_indptr = torch.arange(0, B + 1, dtype = torch.int32, device = "cuda")
    kv_indices = torch.arange(0, B, dtype = torch.int32, device = "cuda")
    kv_last_page_len = torch.full((B, ), S, dtype = torch.int32, device = "cuda")

    decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
      num_qo_heads = N, num_kv_heads = N, head_dim = H, page_size = S)

    no_contention_results = no_contention_greenctx_decodes(decode_wrapper, B)
    all_batch_results.append(no_contention_results)

  df = pd.DataFrame(all_batch_results)
  csv_filename = "greenctx_no_contention_decode_itl" + "_d" + str(D) + ".csv"
  df.to_csv(csv_filename, index = False)

def run_prefill_only_exp():

  all_batch_results = []

  no_contention_results = no_contention_greenctx_prefill()
  all_batch_results.append(no_contention_results)

  df = pd.DataFrame(all_batch_results)
  csv_filename = "greenctx_no_contention_prefill_tp" + "_d" + str(D) + ".csv"
  df.to_csv(csv_filename, index = False)

def main():
#  run_decode_only_exp()
  run_prefill_only_exp()

if __name__ == "__main__":
  main()
