import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import argparse
from matplotlib.lines import Line2D

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
  df1 = pd.read_csv('data/greenctx_no_contention_decode_itl_d8192.csv')
  df2 = pd.read_csv('data/greenctx_contention_decode_serial_prefill_itl_d8192.csv')

  fig, ax = plt.subplots()

  for batch in [128, 512]:
    df1_batch = df1[df1['Batch'] == batch]
    df2_batch = df2[df2['Batch'] == batch]
    ax.plot(df1_batch['Active_SMs'], df1_batch['ITL_ms'], marker = 'o', linestyle = '-', label = 'B: ' + str(batch) + ' no prefill')
    ax.plot(df2_batch['Active_SMs'], df2_batch['ITL_ms'], marker = 'o', linestyle = ':', label = 'B: ' + str(batch) + ' with prefill')

  ax.set_xlabel('Num of SMs')
  ax.set_ylabel('Single layer decode time (ms)')
  ax.legend(title = 'Batch size')

  plt.savefig('plots/delta_decode_serial_prefill_d8192.png')

def plot_delta_decode_itl():
  df1 = pd.read_csv('data/greenctx_no_contention_decode_itl_d8192.csv')
  df2 = pd.read_csv('data/greenctx_contention_decode_serial_prefill_itl_d8192.csv')
  df3 = pd.read_csv('data/greenctx_contention_decode_batched_prefill_itl_d8192.csv')
#  active_sms = {'80/20': 108, '60/40': 76, '40/60': 52}
#  colors = {'80/20': "tab:blue", '60/40': "tab:green", '40/60': "tab:purple"}

  active_sms = {'90/10': 116, '80/20': 108, '70/30': 92, '60/40': 76}
  colors = {'90/10': "tab:blue", '80/20': "tab:green", '70/30': "tab:purple", '60/40': "tab:red"}

  fig, ax = plt.subplots()

  for sm_key in active_sms.keys():
    sm = active_sms[sm_key]
    batches = df1['Batch'].unique()
    x = df1[df1['Active_SMs'] == sm]['ITL_ms'] 
    x.reset_index(drop = True, inplace = True)
    y = df2[df2['Active_SMs'] == sm]['ITL_ms'] 
    y.reset_index(drop = True, inplace = True)
    z = df3[df3['Active_SMs'] == sm]['ITL_ms'] 
    z.reset_index(drop = True, inplace = True)
    delta_itl_percent1 = ((y - x) * 100)/ x
    delta_itl_percent2 = ((z - x) * 100)/ x

    c = colors[sm_key]
    ax.plot(batches[:-1], delta_itl_percent1[:-1], marker = 'o', color = c, linestyle = '-')
    ax.plot(batches[:-1], delta_itl_percent2[:-1], marker = 'o', color = c, linestyle = ':')
  
#    for (batch, delta1, delta2) in zip(batches, delta_itl_percent1, delta_itl_percent2):
#      if (batch != batches[0]):
#        del_delta = delta2 - delta1
#        ax.annotate(f'(+{del_delta:.1f}%)', (batch, delta2), textcoords = 'offset points', xytext = (0, 10), fontsize = 7, color = 'gray')

  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()
  ax.set_xlim(xmin, xmax + 40)      # room for the right-edge labels
  ax.set_ylim(ymin, ymax + 2)       # room for the top labels

  ax.set_xlabel('Batch size')
  ax.set_ylabel('ITL increase %')

  handles = [Line2D([0], [0], color=c, lw=2, label=split) for split, c in colors.items()]
  ax.legend(handles=handles, title="SM split", loc='upper left', bbox_to_anchor=(1.02, 1))

  plt.savefig('plots/delta_itl_serial_batched_prefill_d8192.png', bbox_inches='tight')

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
