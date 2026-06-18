import pandas as pd
from results_io import generate_csv_header_sidecar
from config import B_dec_sweep, S_dec, D, N, H, F, S_prefill_sweep, B_prefill
import model
import benchmark

def run_decode_isolated_experiment():

  all_batch_results = []
  for B in B_dec_sweep: # batch size
    no_contention_results = benchmark.measure_decode_isolated(B)
    all_batch_results.append(no_contention_results)

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])
  
  params = {"B_dec_sweep": B_dec_sweep, "S_dec": S_dec, "D": D, "num_kv_heads": N, "H": H, 
            "F": F, "exp": "decode_only_no_contention"}

  generate_csv_header_sidecar(df, params)

def run_decode_vs_prefill_experiment(prefill_fn):

  all_batch_results = []

  for S_prefill in S_prefill_sweep:
    for B_dec in B_dec_sweep: # batch size
      contention_decode_results = benchmark.measure_decode_under_prefill_contention(prefill_fn, B_dec, B_prefill, S_prefill)
      all_batch_results.append(contention_decode_results)

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])

  params = {"B_dec_sweep": B_dec_sweep, "S_dec": S_dec, "S_prefill_sweep": S_prefill_sweep, "D": D, 
            "num_kv_heads": N, "H": H, "F": F, "prefill_fn": prefill_fn.__name__, 
            "exp": f"decode_with_prefill_contention_{prefill_fn.__name__}"}

  generate_csv_header_sidecar(df, params)

def main():
  # Decode with no prefill contention; varying batch size, partition configs 
  run_decode_isolated_experiment()

  # Decode contention with serial prefill in other green context; varying prefill context length
  run_decode_vs_prefill_experiment(model.do_serial_prefill)

  # Decode contention with batched prefill in other green context; fixed prefill context length 
  run_decode_vs_prefill_experiment(model.do_batched_prefill)

if __name__ == "__main__":
  main()
