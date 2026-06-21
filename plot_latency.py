import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D
import os
import sys

def plot_decode_itl(csv_filename):

  df = pd.read_csv(csv_filename, comment = '#')

  batches = df['Batch'].unique()

  _, ax = plt.subplots()

  for batch in batches:
    df_batch = df[df['Batch'] == batch]
    ax.plot(df_batch['Active_SMs'], df_batch['ITL_ms'], marker = 'o', linestyle = '-', label = 'B: ' + str(batch))

  ax.set_xlabel('Num of SMs')
  ax.set_ylabel('Single layer decode time (ms)')
  ax.legend(title = 'Batch size')

  target_filename = os.path.join('plots', os.path.relpath(csv_filename, 'data'))
  target_filename = target_filename.replace('.csv', '.png')
  os.makedirs(os.path.dirname(target_filename), exist_ok=True)
  plt.savefig(target_filename)
  plt.close()

def plot_decode_vs_prefill_itl_delta(csv_filename1, csv_filename2):
  df_isol = pd.read_csv(csv_filename1, comment = '#')
  df_with_contention = pd.read_csv(csv_filename2, comment = '#')

  S_prefill_list = df_with_contention['S_prefill'].unique()
  active_sms_list = df_with_contention['Active_SMs'].unique()[0:4]
  batches = df_isol['Batch'].unique()[1:]
  markers = ['o', '^', 's']
  marker_map = {S: markers[i] for i, S in enumerate(S_prefill_list)}

  cmap = plt.get_cmap('tab10')
  color_map = {active_sm: cmap(i) for i, active_sm in enumerate(active_sms_list)}


  fig, axs = plt.subplots(ncols=2, nrows=2,
                        layout="constrained")
  axs = axs.flatten()
  # NOTE: We are excluding B = 32 because ITL flat across SM count; decode is launch/latency bound; comparison is not meaningful.
  flag = False
  for i, active_sm in enumerate(active_sms_list):
    ax = axs[i]
    isol = df_isol[(df_isol['Active_SMs'] == active_sm) & (df_isol['Batch'] != 32)]

    for S_prefill in S_prefill_list:
      cont = df_with_contention[(df_with_contention['S_prefill'] == S_prefill) &
                                (df_with_contention['Active_SMs'] == active_sm) &
                                (df_with_contention['Batch'] != 32)]
      m = isol.merge(cont, on=['Batch'], suffixes=('_isol', '_cont')).sort_values('Batch')
      delta = ((m['ITL_ms_cont']- m['ITL_ms_isol']) * 100)/ m['ITL_ms_isol']
      if active_sm == active_sms_list[0]:
        ax.plot(m['Batch'], delta, marker = 'o', linestyle = '-', 
                label = f'S = {S_prefill}')
      else: 
        ax.plot(m['Batch'], delta, marker = 'o', linestyle = '-')

    ax.set_title(f'Active SMs: {active_sm}')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('ITL increase %')

  fig.legend(title = 'S_prefill', loc = 'outside upper right')
  #handles = [Line2D([0], [0], color=c, lw=2, label=split) for split, c in color_map.items()]
  #ax.legend(handles=handles, title='Active SMs', loc='upper left')

  target_filename = os.path.join('plots', os.path.relpath(csv_filename2, 'data'))
  target_filename = target_filename.replace('.csv', '_comparison.png')
  os.makedirs(os.path.dirname(target_filename), exist_ok=True)
  plt.savefig(target_filename)
  plt.close()

def main():
  if (len(sys.argv) == 2):
    plot_decode_itl(sys.argv[1])
  elif (len(sys.argv) == 3):
    plot_decode_vs_prefill_itl_delta(sys.argv[1], sys.argv[2]) 

if __name__ == "__main__":
  main() 
