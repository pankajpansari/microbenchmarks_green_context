"""Extract L2 cache hit rates per NVTX domain from contention NCU CSV files.

For each contention_*_smN.csv in data/, computes:
    L2 hit rate = lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum
                  / lts__t_sectors_srcunit_tex_op_read.sum

Results are reported per domain (prefill / decode) across all SM counts.
The prefill-decode contention range is excluded.

Usage:
    python3 l2_hit_rate.py [--data-dir data/]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

_HIT_METRIC  = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
_READ_METRIC = "lts__t_sectors_srcunit_tex_op_read.sum"
_DOMAINS     = ("prefill", "decode")


def _load_csv(path: Path) -> pd.DataFrame:
    with open(path) as f:
        lines = f.readlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None
    )
    if header_idx is None:
        raise ValueError(f"{path}: no kernel data (no 'ID' header found)")
    df = pd.read_csv(path, skiprows=header_idx)
    df["Metric Value"] = pd.to_numeric(
        df["Metric Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    return df


def _nvtx_domain(thread_col: str) -> str | None:
    """Return 'prefill', 'decode', 'prefill-decode', or None."""
    m = re.search(r'"<default domain>:([^:]+):', thread_col)
    return m.group(1) if m else None


def l2_hit_rates(path: Path) -> dict[str, float]:
    """Return {domain: hit_rate_pct} for prefill and decode."""
    df = _load_csv(path)

    domain_col = df.columns[4]  # thread Domain:Push/Pop_Range:...
    df["_domain"] = df[domain_col].apply(_nvtx_domain)

    rates = {}
    for domain in _DOMAINS:
        sub = df[df["_domain"] == domain]
        hit  = sub.loc[sub["Metric Name"] == _HIT_METRIC,  "Metric Value"].sum()
        total = sub.loc[sub["Metric Name"] == _READ_METRIC, "Metric Value"].sum()
        rates[domain] = (hit / total * 100) if total > 0 else float("nan")
    return rates


def main():
    p = argparse.ArgumentParser(description="L2 hit rates from contention NCU CSVs")
    p.add_argument("--data-dir", default="data", help="directory containing CSVs")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    csvs = sorted(data_dir.glob("contention_*.csv"))
    if not csvs:
        print(f"No contention_*.csv files found in {data_dir}")
        return

    records = []
    for csv_path in csvs:
        sm_m = re.search(r"_sm(\d+)\.csv$", csv_path.name)
        d_m  = re.search(r"_D(\d+)_", csv_path.name)
        num_sms = int(sm_m.group(1)) if sm_m else -1
        d_val   = int(d_m.group(1))  if d_m  else -1
        try:
            rates = l2_hit_rates(csv_path)
        except ValueError as e:
            print(f"Skipping {csv_path.name}: {e}")
            continue
        records.append({"D": d_val, "num_sms": num_sms, **rates})

    result = (
        pd.DataFrame(records)
        .sort_values(["D", "num_sms"])
        .reset_index(drop=True)
    )

    result = result.rename(columns={
        "prefill": "prefill_l2_hit%",
        "decode":  "decode_l2_hit%",
    })

    with pd.option_context("display.float_format", "{:.2f}".format):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
