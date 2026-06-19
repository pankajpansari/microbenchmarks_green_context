import pandas as pd
from results_io import generate_csv_header_sidecar
from config import B_dec_sweep, S_dec, S_prefill_sweep, B_prefill, D_sweep, N
import model
import benchmark
import logging, os, datetime

def run_decode_isolated_experiment(mdl):

  print(f">>>Running decode-only experiment")
  all_batch_results = []

  for B_dec in B_dec_sweep: # batch size
    no_contention_results = benchmark.measure_decode_isolated(mdl, B_dec)
    all_batch_results.append(no_contention_results)
    print(f"-----Completed model D = {mdl.D}, batch size = {B_dec}")

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])
  
  params = {"B_dec_sweep": B_dec_sweep, "S_dec": S_dec, "D": mdl.D, "num_kv_heads": mdl.N, "H": mdl.H, 
            "F": mdl.F, "exp": "decode_only_no_contention"}

  generate_csv_header_sidecar(df, params)

def run_decode_vs_prefill_experiment(mdl,prefill_fn):

  print(f">>>Running decode-vs-prefill experiment")
  all_batch_results = []

  for S_prefill in S_prefill_sweep:
    for B_dec in B_dec_sweep: # batch size
      contention_decode_results = benchmark.measure_decode_under_prefill_contention(mdl, prefill_fn, B_dec, B_prefill, S_prefill)
      all_batch_results.append(contention_decode_results)
      print(f"-----Completed model D = {mdl.D}, prefill context length = {S_prefill}, decode batch size = {B_dec}") 

  df = pd.DataFrame([row for sub in all_batch_results for row in sub])

  params = {"B_dec_sweep": B_dec_sweep, "B_prefill": B_prefill, "S_dec": S_dec, "S_prefill_sweep": S_prefill_sweep, "D": mdl.D, 
            "num_kv_heads": mdl.N, "H": mdl.H, "F": mdl.F, "prefill_fn": prefill_fn.__name__, 
            "exp": f"decode_with_prefill_contention_{prefill_fn.__name__}"}

  generate_csv_header_sidecar(df, params)

def main():
  os.makedirs("data", exist_ok=True)
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
  log_path = os.path.join("data", f"run_{stamp}.log")
  logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.WARNING,
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

  for D in D_sweep:
    print(f"Running experiments for model D sweep: {D_sweep}")
    mdl = model.Model(D, N)

    # Decode with no prefill contention; varying batch size, partition configs 
    run_decode_isolated_experiment(mdl)

    # Decode contention with serial prefill in other green context; varying prefill context length
    run_decode_vs_prefill_experiment(mdl, mdl.do_serial_prefill)

    # Decode contention with batched prefill in other green context; fixed prefill context length 
    run_decode_vs_prefill_experiment(mdl, mdl.do_batched_prefill)
    print(f"Completed experiments for model D sweep: {D_sweep}")

if __name__ == "__main__":
  main()
