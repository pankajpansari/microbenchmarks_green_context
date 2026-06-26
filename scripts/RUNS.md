# Profiling runs

Runners live in `scripts/`. Each is parametrized (configs via flags, not
hardcoded). Source of shared env is `scripts/env.sh` — runners forward the
venv/CUDA paths into `sudo ncu`/`nsys`. Per-machine: export `PROFILE_PATH` /
`PROFILE_PYTHONPATH` / `PROFILE_LD_LIBRARY_PATH` before invoking if your venv or
CUDA libs aren't on the interactive PATH.

This file is the provenance log: what each experiment is, why the knobs are set
the way they are, and where output lands. Add a note here when you add a runner
or do a notable run — keep executable config in the runners, not in comments.

## Runners

| Script | What it measures |
|--------|------------------|
| `prefill_bw_sweep.sh` | Prefill HBM bandwidth: serial vs batched, swept over green-ctx SM count and model dim. Template runner. |
| `contention.sh`       | Prefill + decode under contention in separate green contexts, swept over decode SM count; cache/read-traffic metrics via ncu range-replay. |

## prefill_bw_sweep.sh

Sweeps `D × {serial(B=1), batched(B=16)} × S × SM-count`, one CSV per combo.

- SM counts: 132 (full GPU, no partition) then jumps of 16 down to 16. Green
  ctx allocates in multiples of 8; 128 is skipped (too close to full).
- Metrics: `dram__throughput.avg.pct_of_peak_sustained_elapsed` (HBM bw, % of
  peak) and `gpu__time_duration.sum` (per-kernel duration, for the
  duration-weighted average).
- NVTX range `prefill-sms-<N>` tags the measured region; warmup kernels fall
  outside it and are excluded by `--nvtx-include` (and dropped again by the
  plotter).
- Output: `data/prefill_bw/d<D>/prefill_<fn>_S<S>_B<B>_D<D>_sm<N>.csv`. D is
  split into per-D dirs so the plotter (groups by fn/S/SM only) doesn't collide
  one D with another.

Machine presets:
- **H100** (default): `--D "4096 8192" --N 32`
- **local A10**: `--D "512 1024" --N 8` (and a smaller `--sms` list)

Plotting: point `main()`'s `data_dir` in `plot_prefill_bw.py` at the per-D dir
and run `python3 plot_prefill_bw.py` (once per D).

> Caveat: `profile_prefill.py` currently has `--prefill-fn` / `--batch` /
> `--prefill-sms` commented out (only the serial-all-SMs path is live).
> Re-enable those args before running this sweep.

## contention.sh

Prefill and decode run concurrently in separate green contexts, swept over the
**config.py workload grid × decode green-ctx SM count**. One CSV per combo.

Workload defaults come from `config.py` (the single source of truth, shared with
the Python analysis code) via the `cfg` helper in `env.sh`. Swept dimensions are
the list values there:

| CLI flag | config.py | role |
|----------|-----------|------|
| `-D` | `D_sweep` | swept |
| `--decode-batch` | `B_dec_sweep` | swept |
| `-S-prefill` | `S_prefill_sweep` | swept |
| `-N` | `N` | scalar |
| `-S-decode` | `S_dec` | scalar |
| `--sms` | — | swept, **required** (not in config) |

Any flag overrides its config default for that run (e.g. `-D "512 1024"`).

For each combo: `balance_iters.py` is run first (outside the profiler —
cuda-event timings under ncu are unreliable) to pick prefill/decode iter counts
that make the two streams overlap in time; those iters are then passed to the
profiled run. ncu uses **range-replay** over the NVTX ranges `prefill/`,
`decode/`, `prefill-decode/`.

- Metrics (read-traffic + L2 hit/miss, edit `METRICS` at the top of the script):
  `dram__bytes_read.sum`, `lts__t_sectors_srcunit_tex_op_read[_lookup_hit/miss].sum`,
  `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`.
- Output: `<out-dir>/<prefix>_D<D>_B<B>_Sp<Sp>_sm<N>.csv` (defaults: `./decode_nvtx_*`;
  override with `--out <prefix>` / `--out-dir <dir>`). The D/B/Sp tags keep
  swept combos from colliding.
- Do **not** pass `--decode-sms` / `--prefill-iters` / `--decode-iters` yourself —
  the script supplies them per combo.
- Provenance: each invocation drops a timestamped
  `<prefix>_manifest_<ts>.txt` in the output dir recording the invocation, the
  resolved grid, git rev (+`dirty`), host, and the exact replayable command +
  balanced iters for every CSV. This is the per-run record that replaces
  `cmd_history.txt`.

Example (config.py supplies D/N/batch/S; only the SM sweep is given):

    scripts/contention.sh --sms "16 32 48 64"
