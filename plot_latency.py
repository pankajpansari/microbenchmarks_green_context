import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
from matplotlib.lines import Line2D
import os
import sys
import glob
import re

from parse_ncu import parse_ncu_csv


def plot_kernel_bw_timeline(csv_filename):
  """Marimekko-style timeline of an ncu profile.

  Lays the kernels out left-to-right in execution order as a row of blocks:
  each block's width is proportional to the kernel's duration (us) and its
  height to the achieved HBM bandwidth (% of peak). So the x-axis is a real
  time axis (cumulative kernel duration) and tall-vs-short shows how well each
  kernel saturates memory bandwidth. Blocks are coloured by kernel label.
  """
  df = parse_ncu_csv(csv_filename)
  df = df[df['duration_us'].notna()].reset_index(drop=True)
  if df.empty:
    raise ValueError(f"{csv_filename}: no kernels with a duration metric to plot.")
  df['bw_pct'] = df['bw_pct'].fillna(0.0)

  # Left edge of each block = cumulative duration of everything before it.
  lefts = df['duration_us'].cumsum().shift(fill_value=0.0)

  # One colour per distinct kernel label (stable order of first appearance).
  labels = list(dict.fromkeys(df['kernel']))
  cmap = plt.get_cmap('tab10')
  color_map = {lab: cmap(i % 10) for i, lab in enumerate(labels)}
  colors = df['kernel'].map(color_map)

  fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 5))
  ax.bar(lefts, df['bw_pct'], width=df['duration_us'], align='edge',
         color=colors, edgecolor='white', linewidth=0.8)

  total = df['duration_us'].sum()
  ax.set_xlim(0, total)
  ax.set_ylim(0, 100)
  ax.set_xlabel('Cumulative kernel duration (us)')
  ax.set_ylabel('Achieved HBM bandwidth (% of peak)')
  ax.set_title(os.path.basename(csv_filename) + f'\ntotal {total:.1f} us over {len(df)} kernels')

  handles = [Patch(color=color_map[lab], label=lab) for lab in labels]
  ax.legend(handles=handles, title='Kernel', loc='upper left',
            bbox_to_anchor=(1.0, 1.0), fontsize=8)

  target_filename = os.path.join('plots', os.path.relpath(csv_filename, 'data'))
  target_filename = target_filename.replace('.csv', '_bw_timeline.png')
  os.makedirs(os.path.dirname(target_filename), exist_ok=True)
  fig.savefig(target_filename, bbox_inches='tight', dpi=150)
  plt.close(fig)
  return target_filename

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
  target_filename = target_filename.replace('.csv', '_delta_comparison.png')
  os.makedirs(os.path.dirname(target_filename), exist_ok=True)
  plt.savefig(target_filename)
  plt.close()

def plot_decode_vs_prefill_itl(csv_filename1, csv_filename2, batches = [256, 512]):
  # Overlay ITL time (ms, per-layer) for no-contention decode and decode with prefill contention (varying prefill sizes)
  # csv_filename1: no contention decode info; csv_filename2: decode with (serial or batched) prefills
  df_isol = pd.read_csv(csv_filename1, comment = '#')

  max_sm = max(df_isol['Active_SMs'])
  df_isol = df_isol[(df_isol['Active_SMs'] != max_sm)]

  df_with_contention = pd.read_csv(csv_filename2, comment = '#')

  S_prefill_list = df_with_contention['S_prefill'].unique()
  active_sms_list = df_with_contention['Active_SMs'].unique()[0:4]

  # NOTE: We are excluding B = 32 because ITL flat across SM count; decode is launch/latency bound; comparison is not meaningful.
  #batches = df_isol['Batch'].unique()[1:]
  batches.sort()

  #Check whether all specified batch sizes are actually present in the csv
  assert(set(batches).issubset(set(df_isol['Batch'].unique())))

  markers = {"no_prefill": 'o', "512": '^', "1024": 's', "2048": '*'}
  cmap = plt.get_cmap('tab10')
  color_map = {b: cmap(i) for i, b in enumerate(batches)}

  fig, ax = plt.subplots()

  for i, batch in enumerate(batches):
    isol = df_isol[df_isol['Batch'] == batch]

    ax.plot(isol['Active_SMs'], isol['ITL_ms'], marker = 'o', color = cmap(i), linestyle = ':')

    for j, S_prefill in enumerate(S_prefill_list):
      cont = df_with_contention[(df_with_contention['S_prefill'] == S_prefill) &
                                (df_with_contention['Batch'] == batch)]
      ax.plot(cont['Active_SMs'], cont['ITL_ms'], marker = markers[str(S_prefill)], color = color_map[batch], linestyle = '-')

  ax.set_xlabel('Active_SMs')
  ax.set_ylabel('Single layer decode time (ms)')

  handles = [Line2D([0], [0], marker=m, color = 'black', linestyle = 'None', label=prefill_len) for prefill_len, m in markers.items()]
  fig.legend(handles = handles, title = 'S_prefill', loc = 'outside upper right')
  handles = [Line2D([0], [0], color = c, linestyle = '-', label= str(batch)) for batch, c in color_map.items()]
  fig.legend(handles = handles, title = 'Batch Size', loc = 'outside right')

  target_filename = os.path.join('plots', os.path.relpath(csv_filename2, 'data'))
  target_filename = target_filename.replace('.csv', '_itl_comparison.png')
  os.makedirs(os.path.dirname(target_filename), exist_ok=True)
  plt.savefig(target_filename)
  plt.close()

def plot_all_decode_vs_prefill_itl(subfolder, batches=[256, 512]):
  # Read csvs from data/<subfolder>, discover d values from filenames, and for each d
  # call plot_decode_vs_prefill_itl once for serial and once for batched prefills.
  # Filename convention:
  #   decode_only_no_contention__d<D>__...csv
  #   decode_with_prefill_contention_do_serial_prefill__d<D>__...csv
  #   decode_with_prefill_contention_do_batched_prefill__d<D>__...csv
  data_dir = os.path.join('data', subfolder)

  def find_csv(pattern, d):
    matches = [f for f in glob.glob(os.path.join(data_dir, pattern))
                if not f.endswith('.meta.json')]
    matches = [f for f in matches if re.search(rf'__d{d}__', os.path.basename(f))]
    assert len(matches) == 1, f'Expected exactly 1 match for {pattern} (d={d}), got {matches}'
    return matches[0]

  # Pick up the distinct d values from the no-contention csv filenames
  no_contention_files = [f for f in glob.glob(os.path.join(data_dir, 'decode_only_no_contention__*.csv'))
                          if not f.endswith('.meta.json')]
  d_values = sorted({int(re.search(r'__d(\d+)__', os.path.basename(f)).group(1))
                      for f in no_contention_files})

  for d in d_values:
    no_contention = find_csv('decode_only_no_contention__*.csv', d)
    serial = find_csv('decode_with_prefill_contention_do_serial_prefill__*.csv', d)
    batched = find_csv('decode_with_prefill_contention_do_batched_prefill__*.csv', d)

    plot_decode_vs_prefill_itl(no_contention, serial, batches=list(batches))
    plot_decode_vs_prefill_itl(no_contention, batched, batches=list(batches))

def main():
  files = sorted(glob.glob('data/results_jarvis_june25/prefill_all_sms/*.csv'))

  for f in files:
      print(f'\n===== {f} =====')
      df = parse_ncu_csv(f)
      print(df[['ID', 'kernel', 'bw_pct', 'duration_us']].to_string(index=False))
      plot_kernel_bw_timeline(f)            # writes plots/.../<name>_bw_timeline.png
#  plot_all_decode_vs_prefill_itl('results_runpod_june21')
  return
  if (len(sys.argv) == 2):
    plot_decode_itl(sys.argv[1])
  elif (len(sys.argv) == 3):
    plot_decode_vs_prefill_itl_delta(sys.argv[1], sys.argv[2]) 

if __name__ == "__main__":
  main() 
