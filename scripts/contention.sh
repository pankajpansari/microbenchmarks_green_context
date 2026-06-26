#!/bin/bash
# Prefill + decode under contention, in separate green contexts, swept over the
# workload grid in config.py x the decode green-ctx SM count. One CSV per combo.
#
# Defaults come from config.py (single source of truth for the experiment grid);
# flags override per run. The swept dimensions are the list values in config.py
# (D_sweep, B_dec_sweep, S_prefill_sweep) crossed with --sms. Per combo, the
# prefill/decode iter counts are balanced by balance_iters.py (run *outside* ncu
# -- cuda-event timings under the profiler are unreliable) so the two streams
# overlap in time; then profiled with ncu range-replay over the nvtx ranges
# prefill/ , decode/ and prefill-decode/.
#
# Usage:
#   scripts/contention.sh --sms "<s ...>" \
#     [-D "<d ...>"] [-N <n>] [--decode-batch "<b ...>"] \
#     [-S-prefill "<s ...>"] [-S-decode <s>] [--out <prefix>] [--out-dir <dir>]
#
# Example (config.py supplies D/N/batch/S; only the SM sweep is given here):
#   scripts/contention.sh --sms "16 32 48 64"
#
# Output: <out-dir>/<prefix>_D<D>_B<B>_Sp<Sp>_sm<N>.csv
#         (default dir: . , default prefix: decode_nvtx)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/env.sh"

# L2/L1 read-traffic + cache hit/miss metrics for the contention study.
METRICS="dram__bytes_read.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"

# ---- defaults from config.py (override with the matching flags) -------------
D_SWEEP="$(cfg D_sweep)"             # swept
N="$(cfg N)"                         # scalar
DEC_BATCH_SWEEP="$(cfg B_dec_sweep)" # swept
S_PREFILL_SWEEP="$(cfg S_prefill_sweep)" # swept
S_DECODE="$(cfg S_dec)"             # scalar
SM_SWEEP=""                         # not in config.py -- must be given via --sms
OUT_PREFIX="decode_nvtx"
OUT_DIR="."

ORIG_ARGS=("$@")  # capture the full invocation before the parse loop consumes it

while [[ $# -gt 0 ]]; do
  case "$1" in
    -D|--dim)            D_SWEEP="$2";        shift 2 ;;
    -N|--heads)          N="$2";              shift 2 ;;
    --decode-batch)      DEC_BATCH_SWEEP="$2"; shift 2 ;;
    -S-prefill)          S_PREFILL_SWEEP="$2"; shift 2 ;;
    -S-decode)           S_DECODE="$2";       shift 2 ;;
    --sms)               SM_SWEEP="$2";       shift 2 ;;
    --out)               OUT_PREFIX="$2";     shift 2 ;;
    --out-dir)           OUT_DIR="$2";        shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$SM_SWEEP" ]]; then
  echo "error: --sms \"<s ...>\" is required (decode green-ctx SM counts to sweep)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "grid: D={$D_SWEEP} N=$N decode-batch={$DEC_BATCH_SWEEP} S-prefill={$S_PREFILL_SWEEP} S-decode=$S_DECODE sms={$SM_SWEEP}"

# Provenance sidecar: one timestamped manifest per invocation, recording what
# produced the CSVs in this dir. Timestamped so reruns don't clobber each other.
GIT_REV=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)
git -C "$REPO" diff --quiet 2>/dev/null || GIT_REV="$GIT_REV (dirty)"
MANIFEST="${OUT_DIR}/${OUT_PREFIX}_manifest_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "# contention.sh run manifest"
  echo "timestamp:  $(date -Iseconds 2>/dev/null || date)"
  echo "host:       $(hostname)"
  echo "git:        $GIT_REV"
  echo "invocation: $0 ${ORIG_ARGS[*]}"
  echo "grid:       D={$D_SWEEP} N=$N decode-batch={$DEC_BATCH_SWEEP} S-prefill={$S_PREFILL_SWEEP} S-decode=$S_DECODE sms={$SM_SWEEP}"
  echo "ncu:        --replay-mode range --nvtx-include prefill/,decode/,prefill-decode/"
  echo "metrics:    $METRICS"
  echo
} > "$MANIFEST"
echo "manifest: $MANIFEST"

for D in $D_SWEEP; do
 for B in $DEC_BATCH_SWEEP; do
  for SP in $S_PREFILL_SWEEP; do
   for sms in $SM_SWEEP; do
    echo "=== D=$D B=$B S-prefill=$SP decode-sms=$sms ==="
    BASE_ARGS=(-D "$D" -N "$N" --decode-batch "$B" -S-prefill "$SP" -S-decode "$S_DECODE")

    # Balance iters for THIS combo, without profiler overhead.
    ITERS=$(python3 "$REPO/balance_iters.py" "${BASE_ARGS[@]}" --decode-sms "$sms")
    PREFILL_ITERS=$(echo "$ITERS" | grep 'prefill_iters=' | cut -d= -f2)
    DECODE_ITERS=$(echo "$ITERS" | grep 'decode_iters=' | cut -d= -f2)
    echo "balanced: prefill_iters=$PREFILL_ITERS decode_iters=$DECODE_ITERS"

    CSV="${OUT_DIR}/${OUT_PREFIX}_D${D}_B${B}_Sp${SP}_sm${sms}.csv"
    PROFILE_CMD=(python3 "$REPO/profile_prefill_decode.py" "${BASE_ARGS[@]}"
      --decode-sms "$sms" --prefill-iters "$PREFILL_ITERS" --decode-iters "$DECODE_ITERS")

    # Record the exact, replayable command for this CSV.
    {
      echo "[$(basename "$CSV")]"
      echo "  balanced: prefill_iters=$PREFILL_ITERS decode_iters=$DECODE_ITERS"
      echo "  cmd: $(printf '%q ' "${PROFILE_CMD[@]}")"
      echo
    } >> "$MANIFEST"

    sudo_env ncu --replay-mode range \
      --nvtx --nvtx-include "prefill/" --nvtx-include "decode/" --nvtx-include "prefill-decode/" \
      --metrics "$METRICS" \
      --csv --log-file "$CSV" \
      "${PROFILE_CMD[@]}"
   done
  done
 done
done

echo "Done. CSVs: ${OUT_DIR}/${OUT_PREFIX}_D<D>_B<B>_Sp<Sp>_sm<N>.csv"
