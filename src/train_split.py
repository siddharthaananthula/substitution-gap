"""Time-based training split for the acceptance model (no leakage from final orders).

Label order    = each customer's LAST PRIOR order (highest prior order_number).
Feature window = all prior orders before it.
Everything the model learns from (regulars, pairs, popularity) is rebuilt from the
feature window with the existing functions. Final (train/test) orders are never read.

Run with: python -m src.train_split
"""
import gc
import time

import pandas as pd

from src.build_substitutes import OUT, EXPECTED, Stage, downcast_counts, peak_mb
from src.evaluate import (
    GROUP_COL,
    TRUTHS,
    add_truth_sizes,
    build_events,
    first_time_buys,
)
from src.load import load_orders, load_prior_lines, load_products
from src.pairs import (
    add_lift_lower,
    add_names,
    adjust_for_exploration,
    compute_lift,
    find_regulars,
    orders_per_user,
    run_batches,
)
from src.rankers import MIN_LIFT_LOW_ADJ, first_buy_counts

WINDOW_DIR = OUT / "train_window"


def split_last_prior(lines):
    """Split prior lines into (feature window, label order lines): label = last prior order."""
    last = lines.groupby("user_id")["order_number"].transform("max")
    is_label = (lines["order_number"] == last).to_numpy()
    window, label = lines[~is_label], lines[is_label]
    assert len(window) + len(label) == len(lines)
    return window, label


def assert_prior_only(order_ids, orders):
    """Fail if any order id belongs to a final (train or test) order."""
    final = orders.loc[orders["eval_set"] != "prior", "order_id"]
    leaked = pd.Index(order_ids).isin(final)
    assert not leaked.any(), f"{leaked.sum()} final-order ids found in training data"


def main():
    t_start = time.perf_counter()

    with Stage("load_prior_lines"):
        lines = load_prior_lines()
    print(f"  prior lines: {len(lines):,}")

    with Stage("split_last_prior"):
        window, label_lines = split_last_prior(lines)
        del lines
        gc.collect()
        orders = load_orders()[["order_id", "eval_set"]]
        assert_prior_only(window["order_id"].unique(), orders)
        assert_prior_only(label_lines["order_id"].unique(), orders)
        del orders
    print(f"  feature window lines: {len(window):,}, label order lines: {len(label_lines):,}, "
          f"label orders: {label_lines['order_id'].nunique():,}")

    with Stage("find_regulars (window)"):
        reg = find_regulars(window, orders_per_user(window))
    print(f"  regular pairs: {len(reg):,} (all prior: {EXPECTED['regular pairs']:,}), "
          f"customers: {reg['user_id'].nunique():,}, products: {reg['product_id'].nunique():,}")

    with Stage("run_batches (window)"):
        totals = run_batches(window, reg, n_batches=10)

    with Stage("first_buy_counts (window)"):
        counts = first_buy_counts(window)
    print(f"  products with first-time buys: {len(counts):,}")
    del window
    gc.collect()

    with Stage("lift (window)"):
        pairs = adjust_for_exploration(add_lift_lower(compute_lift(totals)), totals)
        products = load_products()
        n_before = len(pairs)
        pairs = downcast_counts(add_names(pairs, products[["product_id", "product_name"]]))
        assert len(pairs) == n_before
    ratio = round(float((pairs["lift"] / pairs["lift_adj"]).iloc[0]), 3)
    del totals

    with Stage("training events"):
        label_lines = label_lines[["order_id", "user_id", "product_id", "reordered"]]
        events = build_events(reg, label_lines)
        truth = first_time_buys(label_lines)
        events["event_id"] = pd.RangeIndex(len(events)).astype("int32")
        groups = {t: products.set_index("product_id")[c] for t, c in GROUP_COL.items()}
        events = add_truth_sizes(events, truth, groups)

    with Stage("save"):
        WINDOW_DIR.mkdir(parents=True, exist_ok=True)
        reg.to_parquet(WINDOW_DIR / "regulars.parquet", index=False)
        pairs.to_parquet(WINDOW_DIR / "substitutes.parquet", index=False)
        counts.rename("count").rename_axis("product_id").reset_index().to_parquet(
            WINDOW_DIR / "first_buy_counts.parquet", index=False)
        events.to_parquet(WINDOW_DIR / "events.parquet", index=False)
        truth.to_parquet(WINDOW_DIR / "truth.parquet", index=False)
    print(f"  saved to {WINDOW_DIR}")

    print(f"\nTotal: {time.perf_counter() - t_start:,.1f}s, peak memory {peak_mb():,.0f} MB")

    strong = pairs[pairs["lift_low_adj"] > MIN_LIFT_LOW_ADJ]
    print("\nFeature window vs all prior orders:")
    print(f"  candidate pairs:            {len(pairs):>9,}  (all prior {EXPECTED['candidate pairs']:,})")
    print(f"  exploration ratio:          {ratio:>9}  (all prior {EXPECTED['exploration ratio']})")
    print(f"  pairs lift_low_adj > 2:     {len(strong):>9,}  "
          f"(all prior {EXPECTED['pairs lift_low_adj > 2']:,})")
    print(f"  regular products with one:  {strong['a'].nunique():>9,}  "
          f"(all prior {EXPECTED['regular products lift_low_adj > 2']:,})")

    print("\nTraining events (label = last prior order):")
    print(f"  events:    {len(events):,}  (evaluation: 65,476)")
    print(f"  customers: {events['user_id'].nunique():,}  (evaluation: 29,701)")
    for t in TRUTHS:
        print(f"  ceiling {t:<6} {(events[f'n_{t}'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
