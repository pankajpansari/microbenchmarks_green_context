import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import argparse

def plot_decode_info(csv_filename, target_filename):
  df = pd.read_csv(csv_filename)

  records = []
  for _, row in df.iterrows():
    for entry in row:
      records.append(ast.literal_eval(entry))

  df_flat = pd.DataFrame(records)

  batches = df_flat['Batch'].unique()

  fig, ax = plt.subplots()

  for batch in batches:
    df_batch = df_flat[df_flat['Batch'] == batch]
    ax.plot(df_batch['Active_SMs'], df_batch['ITL_ms'], marker = 'o', linestyle = '-', label = 'B: ' + str(batch))

  ax.set_xlabel('Num of SMs')
  ax.set_ylabel('Single layer decode time (ms)')
  ax.legend(title = 'Batch size')

  plt.savefig(target_filename)

def plot_delta_decode_info():
  df1 = pd.read_csv('data/greenctx_no_contention_decode_itl_d4096.csv')
  df2 = pd.read_csv('data/greenctx_contention_decode_serial_prefill_itl_d4096.csv')

  records = []
  for _, row in df1.iterrows():
    for entry in row:
      records.append(ast.literal_eval(entry))

  df1_flat = pd.DataFrame(records)

  records = []
  for _, row in df2.iterrows():
    for entry in row:
      records.append(ast.literal_eval(entry))

  df2_flat = pd.DataFrame(records)

  fig, ax = plt.subplots()

  for batch in [128, 512]:
    df1_batch = df1_flat[df1_flat['Batch'] == batch]
    df2_batch = df2_flat[df2_flat['Batch'] == batch]
    ax.plot(df1_batch['Active_SMs'], df1_batch['Elapsed_time_ms'], marker = 'o', linestyle = '-', label = 'B: ' + str(batch) + ' no prefill')
    ax.plot(df2_batch['Active_SMs'], df2_batch['ITL_ms'], marker = 'o', linestyle = ':', label = 'B: ' + str(batch) + ' with prefill')

  ax.set_xlabel('Num of SMs')
  ax.set_ylabel('Single layer decode time (ms)')
  ax.legend(title = 'Batch size')

  plt.savefig('plots/delta_decode_serial_prefill_d4096.png')

def plot_delta_decode_itl():
  df1 = pd.read_csv('data/greenctx_no_contention_decode_itl_d8192.csv')
  df2 = pd.read_csv('data/greenctx_contention_decode_serial_prefill_itl_d8192.csv')
  active_sms = {'80/20': 108, '60/40': 76, '40/60': 52}

  fig, ax = plt.subplots()

  for sm_key in active_sms.keys():
    sm = active_sms[sm_key]
    batches = df1['Batch'].unique()
    x = df1[df1['Active_SMs'] == sm]['ITL_ms'] 
    x.reset_index(drop = True, inplace = True)
    y = df2[df2['Active_SMs'] == sm]['ITL_ms'] 
    y.reset_index(drop = True, inplace = True)
    delta_itl_percent = ((y - x) * 100)/ x
    ax.plot(batches, delta_itl_percent, marker = 'o', linestyle = '-', label = sm_key + ' prefill')
  
  ax.set_xlabel('Batch size')
  ax.set_ylabel('ITL increase %')
  ax.legend(title = 'SM split')

  plt.savefig('plots/delta_itl_serial_prefill_d8192.png')

def plot_prefill_info(csv_filename, target_filename):
  df = pd.read_csv(csv_filename)

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

  plt.savefig(target_filename, bbox_inches = 'tight')

def main():
#  parser = argparse.ArgumentParser()
#  parser.add_argument('data_filename')
#  parser.add_argument('plot_filename')
#  args = parser.parse_args()
#  plot_decode_info(args.data_filename, args.plot_filename)
#  plot_delta_decode_info()
  plot_delta_decode_itl()

if __name__ == "__main__":
  main() 
