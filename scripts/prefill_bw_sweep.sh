#!/bin/bash
# Prefill HBM-bandwidth sweep: serial vs batched prefill, across green-ctx SM
# counts, for one or more model dims. One CSV per (fn, S, B, D, SM).
#
# This is the template runner: it defines the *sweep machinery*; the configs
# come from flags (with H100 defaults baked in). You should never need to edit
# the body to change what runs -- pass flags instead. e.g.
#
#   bash scripts/prefill_bw_sweep.sh                       # full H100 sweep
#   bash scripts/prefill_bw_sweep.sh --D 512 --N 8 \       # local A10
#        --sms 72 --configs "serial:512:1 batched:2048:16"
#
# Metrics:
#   dram__throughput.avg.pct_of_peak_sustained_elapsed  -> HBM bw, % of peak
#   gpu__time_duration.sum                              -> per-kernel duration
#                                                          (duration-weighted avg)
# NVTX range "prefill-sms-<N>" tags the measured region; warmups fall outside it
# and are dropped by --nvtx-include (and again by the plotter).
#
# NOTE: requires profile_prefill.py to expose --prefill-fn / --batch /
# --prefill-sms (currently commented out in that file -- re-enable before use).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/env.sh"

# ---- defaults (H100: N=32, matching plots/results_runpod_june21) ------------
D_SWEEP="4096 8192"
N=32
# SM counts: 132 (full GPU) then jumps of 16. Green ctx allocates in multiples
# of 8; 128 skipped (too close to full).
SM_SWEEP="132 112 96 80 64 48 32 16"
# Each config is fn:S:B  (serial implies B=1).
CONFIGS="serial:512:1 serial:2048:1 batched:512:16 batched:2048:16"
OUT_ROOT="$REPO/data/prefill_bw"

usage() {
  grep '^#' "$0" | sed 's/^# \?//'
  cat <<EOF

Flags (all optional):
  --D    "<d ...>"     model dims to sweep        (default: $D_SWEEP)
  --N    <n>           num attention heads        (default: $N)
  --sms  "<s ...>"     green-ctx SM counts        (default: $SM_SWEEP)
  --configs "<fn:S:B ...>"  prefill configs       (default: $CONFIGS)
  --out  <dir>         output root                (default: $OUT_ROOT)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --D)       D_SWEEP="$2"; shift 2 ;;
    --N)       N="$2"; shift 2 ;;
    --sms)     SM_SWEEP="$2"; shift 2 ;;
    --configs) CONFIGS="$2"; shift 2 ;;
    --out)     OUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

# D split into per-D dirs so the plotter (which groups only by fn/S/SM) does not
# collide one D with another.
for D in $D_SWEEP; do
  OUT="$OUT_ROOT/d${D}"
  mkdir -p "$OUT"
  for spec in $CONFIGS; do
    fn=${spec%%:*}                 # serial | batched
    rest=${spec#*:}; S=${rest%%:*} # prefill context length
    B=${rest##*:}                  # prefill batch size
    for sms in $SM_SWEEP; do
      echo ">> D=$D fn=$fn S=$S B=$B sms=$sms"
      sudo_env ncu --nvtx --nvtx-include "prefill-sms-${sms}/" \
        --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
        --csv --log-file "${OUT}/prefill_${fn}_S${S}_B${B}_D${D}_sm${sms}.csv" \
        python3 "$REPO/profile_prefill.py" \
          -D "$D" -N "$N" --prefill-fn "$fn" -S "$S" --batch "$B" --prefill-sms "$sms"
    done
  done
done

echo "Done. CSVs under $OUT_ROOT/d<D>/"
