import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

def plot_decode_info():
  df = pd.read_csv('greenctx_no_contention_decode_itl_d8192.csv')

  records = []
  for _, row in df.iterrows():
    for entry in row:
      records.append(ast.literal_eval(entry))

  df_flat = pd.DataFrame(records)

  batches = df_flat['Batch'].unique()

  fig, ax = plt.subplots()

  for batch in batches:
    df_batch = df_flat[df_flat['Batch'] == batch]
    ax.plot(df_batch['Active_SMs'], df_batch['Elapsed_time_ms'], marker = 'o', linestyle = '-', label = 'B: ' + str(batch))

  ax.set_xlabel('Num of SMs')
  ax.set_ylabel('Single layer decode time (ms)')
  ax.legend(title = 'Batch size')
  ax.set_title('Decode in green context without contention')
  plt.savefig('no-contention-decodes-green-ctx-all-batches-d8192-h100.png')

def plot_prefill_info():
  df = pd.read_csv('greenctx_no_contention_batched_serial_prefill_tp_d8192.csv')

#  records = []
#  for _, row in df.iterrows():
#    for entry in row:
#      records.append(ast.literal_eval(entry))
#
#  df_flat = pd.DataFrame(records)

  df_flat = df
  fig, ax = plt.subplots()

  ax.plot(df_flat['Active_SMs'], df_flat['Throughput_toks_s'], marker = 'o', linestyle = '-')

  ax.set_xlabel('Num of SMs')
  ax.set_ylabel('Single layer prefill throughput (toks/s)')
  ax.set_title('Prefill in green context without contention')
  plt.savefig('no-contention-batched-serial-prefill-green-ctx-d8192-h100.png', bbox_inches = 'tight')

def main():
  plot_prefill_info()

if __name__ == "__main__":
  main() 
