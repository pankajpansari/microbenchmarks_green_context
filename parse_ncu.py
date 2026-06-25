"""Parse an ncu --csv profile into a tidy per-kernel DataFrame.

ncu writes a "long" CSV: one row per (kernel instance x metric), preceded by a
few ==PROF== banner lines. This collapses it to one row per kernel instance with
the metrics we care about as columns:

    kernel        - short, readable kernel label
    bw_pct        - dram__throughput.avg.pct_of_peak_sustained_elapsed  (% of peak HBM bw)
    duration_us   - gpu__time_duration.sum                              (kernel duration, us)
    kernel_full   - the raw mangled kernel name

Older CSVs that only logged the bandwidth metric still parse fine; duration_us
is just NaN there.

Usage:
    python3 parse_ncu.py data/decode_all_sms/decode_only_D512_S512_B512.csv
    # or import:  from parse_ncu import parse_ncu_csv
"""

import argparse
import pandas as pd

# ncu metric name -> output column
_METRIC_COLS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "bw_pct",
    "gpu__time_duration.sum": "duration_us",  # normalised to us via its unit
}

# (substring in mangled name) -> friendly label, first match wins
_KERNEL_LABELS = [
    ("MergeStates", "merge_states (flashinfer)"),
    ("Decode", "decode_attn (flashinfer)"),
    ("Prefill", "prefill_attn (flashinfer)"),
    ("nvjet", "linear ops"),
    ("gemm", "gemm (cutlass)"),
    ("launch_clamp_scalar", "relu/clamp"),
    ("direct_copy_kernel", "copy"),
    ("elementwise_kernel", "elementwise"),
]


def _short_kernel(name: str) -> str:
    for needle, label in _KERNEL_LABELS:
        if needle in name:
            return label
    # fallback: strip template args / namespaces for something readable
    base = name.split("(")[0]
    return base.split("::")[-1] or name


def _to_us(value: float, unit: str) -> float:
    """Normalise a duration metric value to microseconds."""
    u = (unit or "").lower()
    if u in ("us", "usecond", "useconds", "microsecond"):
        return value
    if u in ("ns", "nsecond", "nseconds", "nanosecond"):
        return value / 1e3
    if u in ("ms", "msecond", "mseconds", "millisecond"):
        return value * 1e3
    if u in ("s", "second", "seconds"):
        return value * 1e6
    return value  # unknown unit: leave as-is


def parse_ncu_csv(path: str) -> pd.DataFrame:
    """Return one row per kernel instance with bw_pct and duration_us columns."""
    # Find the real header row (skip the ==PROF== banner).
    with open(path) as f:
        lines = f.readlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if header_idx is None:
        raise ValueError(
            f"{path}: no kernel data found "
            "(ncu likely reported 'No kernels were profiled')."
        )

    df = pd.read_csv(path, skiprows=header_idx)
    # ncu may write large values with digit-group separators (e.g. "32,128" or
    # the Indian-grouped "85,63,360"); strip commas before parsing to numbers.
    df["Metric Value"] = pd.to_numeric(
        df["Metric Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    rows = []
    for kid, g in df.groupby("ID", sort=True):
        full = g["Kernel Name"].iloc[0]
        row = {"ID": kid, "kernel": _short_kernel(full), "kernel_full": full}
        for _, r in g.iterrows():
            col = _METRIC_COLS.get(r["Metric Name"])
            if col is None:
                continue
            val = r["Metric Value"]
            if col == "duration_us":
                val = _to_us(val, r["Metric Unit"])
            row[col] = val
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in ("bw_pct", "duration_us"):
        if col not in out:
            out[col] = pd.NA
    return out[["ID", "kernel", "bw_pct", "duration_us", "kernel_full"]]


def main():
    p = argparse.ArgumentParser(description="Parse an ncu --csv file into a per-kernel table.")
    p.add_argument("csv", help="path to an ncu --csv log file")
    p.add_argument("-o", "--out", help="optional path to write the parsed table as CSV")
    args = p.parse_args()

    df = parse_ncu_csv(args.csv)

    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 40):
        print(df.drop(columns="kernel_full"))

    if df["duration_us"].notna().any():
        w = df["duration_us"].fillna(0)
        wavg = (df["bw_pct"] * w).sum() / w.sum() if w.sum() else float("nan")
        print(f"\nDuration-weighted avg HBM bw: {wavg:.2f}% of peak "
              f"over {len(df)} kernels, total {w.sum():.1f} us")
    else:
        print("\n(no duration metric in this CSV -> only per-kernel bw_pct available)")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
