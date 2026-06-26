#!/bin/bash
set -e

# Usage: run_contention.sh -D <dim> -N <heads> --decode-batch <B> \
#          -S-prefill <S> -S-decode <S> --decode-sms <sms> [--ncu <out_file>] [--nsys <out_file>]

NCU_OUT=""
NSYS_OUT=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ncu)  NCU_OUT="$2";  shift 2 ;;
    --nsys) NSYS_OUT="$2"; shift 2 ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

# Balance iters without profiler overhead
ITERS=$(python3 balance_iters.py "${PASS_ARGS[@]}")
PREFILL_ITERS=$(echo "$ITERS" | grep 'prefill_iters=' | cut -d= -f2)
DECODE_ITERS=$(echo "$ITERS" | grep 'decode_iters=' | cut -d= -f2)

echo "Balanced: prefill_iters=$PREFILL_ITERS decode_iters=$DECODE_ITERS"

PROFILE_CMD=(python3 profile_prefill_decode.py "${PASS_ARGS[@]}"
  --prefill-iters "$PREFILL_ITERS"
  --decode-iters "$DECODE_ITERS")

if [[ -n "$NCU_OUT" ]]; then
  sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    ncu --replay-mode application --nvtx --nvtx-include "prefill-decode/" \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
    --csv --log-file "$NCU_OUT" \
    "${PROFILE_CMD[@]}"
elif [[ -n "$NSYS_OUT" ]]; then
  sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    nsys profile -t cuda,nvtx \
    --output "$NSYS_OUT" --force-overwrite true \
    "${PROFILE_CMD[@]}"
else
  "${PROFILE_CMD[@]}"
fi
