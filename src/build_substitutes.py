"""Build the full substitute-pair table and save it to data/processed.

Run with: python -m src.build_substitutes
"""
import gc
import time
from pathlib import Path

import numpy as np
import psutil

from src.load import load_prior_lines, load_products
from src.pairs import (
    add_lift_lower,
    add_names,
    adjust_for_exploration,
    compute_lift,
    find_regulars,
    orders_per_user,
    run_batches,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

# From the earlier full-data run (DECISIONS.md D015)
EXPECTED = {
    "regular pairs": 223_366,
    "candidate pairs": 209_725,
    "exploration ratio": 1.229,
    "pairs lift_low_adj > 2": 423,
    "regular products lift_low_adj > 2": 231,
}


def peak_mb():
    """Peak memory of this process so far, in MB (peak working set on Windows)."""
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e6


class Stage:
    """Context manager that prints elapsed time and peak memory for a stage."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        print(f"[{self.name}] {dt:,.1f}s, peak memory {peak_mb():,.0f} MB")


def check(name, actual):
    """Print actual vs expected, flagging any mismatch."""
    exp = EXPECTED[name]
    ok = round(actual, 3) == exp if isinstance(exp, float) else actual == exp
    print(f"  {name:<36} {actual:>12,} expected {exp:>10,}  {'OK' if ok else 'MISMATCH'}")
    return ok


def downcast_counts(pairs, cols=("k_m", "n_m", "n_b", "k_b")):
    """Cast count columns to int32, failing loudly if any value would change."""
    lo, hi = np.iinfo("int32").min, np.iinfo("int32").max
    out = pairs.copy()
    for c in cols:
        col = out[c]
        if col.isna().any() or (col % 1 != 0).any():
            raise ValueError(f"{c} has missing or non-whole values")
        if col.min() < lo or col.max() > hi:
            raise ValueError(f"{c} is outside the int32 range")
        out[c] = col.astype("int32")
    return out


def main():
    t_start = time.perf_counter()

    with Stage("load_prior_lines"):
        lines = load_prior_lines()
    print(f"  prior lines: {len(lines):,}")

    with Stage("orders_per_user"):
        n_orders = orders_per_user(lines)
    print(f"  customers: {len(n_orders):,}")

    with Stage("find_regulars"):
        reg = find_regulars(lines, n_orders)
    print(f"  regular pairs: {len(reg):,}, customers: {reg['user_id'].nunique():,}, "
          f"products: {reg['product_id'].nunique():,}")

    with Stage("run_batches"):
        totals = run_batches(lines, reg, n_batches=10)
    for key, s in totals.items():
        print(f"  {key}: {len(s):,} rows, total {s.sum():,}")
    del lines, n_orders
    gc.collect()

    with Stage("compute_lift"):
        pairs = compute_lift(totals)
    print(f"  candidate pairs: {len(pairs):,}")

    with Stage("add_lift_lower"):
        pairs = add_lift_lower(pairs)
    print(f"  rows: {len(pairs):,}")

    with Stage("adjust_for_exploration"):
        pairs = adjust_for_exploration(pairs, totals)
    print(f"  rows: {len(pairs):,}")
    ratio = round(float((pairs["lift"] / pairs["lift_adj"]).iloc[0]), 3)

    with Stage("add_names"):
        products = load_products()[["product_id", "product_name"]]
        n_before = len(pairs)
        pairs = add_names(pairs, products)
        assert len(pairs) == n_before, "row count changed in add_names"
    print(f"  rows: {len(pairs):,}, missing names: "
          f"{int(pairs['a_name'].isna().sum() + pairs['b_name'].isna().sum())}")

    with Stage("save parquet"):
        pairs = downcast_counts(pairs)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "substitutes.parquet"
        pairs.to_parquet(path, index=False)
    print(f"  saved {len(pairs):,} pairs to {path}")

    print(f"\nTotal: {time.perf_counter() - t_start:,.1f}s, "
          f"peak memory {peak_mb():,.0f} MB")

    n_cand_a = pairs["a"].nunique()
    print(f"\nlift_low_adj thresholds (of {n_cand_a:,} regular products with candidate "
          f"pairs; {reg['product_id'].nunique():,} regular products overall):")
    for cut in (1, 2):
        sel = pairs[pairs["lift_low_adj"] > cut]
        print(f"  > {cut}: {len(sel):,} pairs, {sel['a'].nunique():,} regular products "
              f"({sel['a'].nunique() / n_cand_a:.0%})")

    print("\nSanity check vs earlier run:")
    strong = pairs[pairs["lift_low_adj"] > 2]
    results = [
        check("regular pairs", len(reg)),
        check("candidate pairs", len(pairs)),
        check("exploration ratio", ratio),
        check("pairs lift_low_adj > 2", len(strong)),
        check("regular products lift_low_adj > 2", strong["a"].nunique()),
    ]
    print("All match." if all(results) else "Some numbers DO NOT match.")


if __name__ == "__main__":
    main()
