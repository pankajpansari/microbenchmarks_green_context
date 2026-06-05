import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import argparse

def plot_decode_info(csv_filename):
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

  png_filename = csv_filename.split('.')[0] + '.png'
  plt.savefig(png_filename)

def plot_prefill_info(csv_filename):
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

  png_filename = csv_filename.split('.')[0] + '.png'
  plt.savefig(png_filename, bbox_inches = 'tight')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('filename')
  args = parser.parse_args()
  plot_decode_info(args.filename)

if __name__ == "__main__":
  main() 
