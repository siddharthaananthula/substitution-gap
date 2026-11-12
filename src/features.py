"""Features for the acceptance model: one row per (event, candidate B).

Training rows use only the training feature window (prior orders before each
customer's last prior order); evaluation rows use all prior orders. Training
features are built inside no_final_orders(), which makes any read of final-order
lines or all-prior artefacts raise.

Most features depend only on (A, B), so they are computed once per pool row and
joined to events; customer features are joined by user_id.

Run with: python -m src.features
"""
import gc
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_substitutes import peak_mb
from src.candidates import side_inputs
from src.evaluate import EVAL_DIR
from src.load import DATA, load_orders, load_prior_lines, load_products
from src.rankers import SUBSTITUTES, name_vectors
from src.train_split import WINDOW_DIR, assert_prior_only, split_last_prior

BEHAVIOUR = ["has_pair", "lift_adj", "lift_low_adj", "log1p_k_m", "log1p_n_m"]
FEATURES = BEHAVIOUR + [
    "name_cosine", "log1p_b_first_buys", "b_aisle_pop_rank",
    "same_aisle", "same_department", "b_reorder_rate",
    "cust_first_buy_share", "cust_n_orders",
    "organic_a", "organic_b", "organic_switch",
]
# NaN is allowed only here: no pair (lift), A without any pair (n_m), B never bought (reorder rate)
NAN_OK = {"lift_adj", "lift_low_adj", "log1p_n_m", "b_reorder_rate"}
KEYS = ["event_id", "user_id", "a", "b"]
FORBIDDEN_FILES = {DATA / "order_products__train.csv", SUBSTITUTES}
FORBIDDEN_DIRS = {EVAL_DIR}


@contextmanager
def no_final_orders():
    """Inside this block, reading final-order lines or all-prior artefacts raises."""
    real = {"read_csv": pd.read_csv, "read_parquet": pd.read_parquet}
    files = {Path(f).resolve() for f in FORBIDDEN_FILES}
    dirs = {Path(d).resolve() for d in FORBIDDEN_DIRS}

    def guard(fn):
        def wrapped(path, *args, **kwargs):
            p = Path(path).resolve()
            if p in files or any(d == p or d in p.parents for d in dirs):
                raise RuntimeError(f"leakage: tried to read {p} while building training features")
            return fn(path, *args, **kwargs)
        return wrapped

    pd.read_csv, pd.read_parquet = guard(real["read_csv"]), guard(real["read_parquet"])
    try:
        yield
    finally:
        pd.read_csv, pd.read_parquet = real["read_csv"], real["read_parquet"]


def line_stats(lines):
    """Product reorder rate and customer exploration (first-buy share, order count)."""
    product = lines.groupby("product_id")["reordered"].mean().astype("float32")
    first = (lines["reordered"] == 0)
    customer = pd.DataFrame({
        "cust_first_buy_share": first.groupby(lines["user_id"]).mean(),
        "cust_n_orders": lines.groupby("user_id")["order_id"].nunique(),
    }).astype("float32")
    return product, customer


def pair_features(pool, pairs, counts, products, X, product_reorder):
    """Features that depend only on (A, B), one row per pool row."""
    p = products.set_index("product_id")
    f = pool[["a", "b"]].copy()

    beh = pairs[["a", "b", "lift_adj", "lift_low_adj", "k_m"]]
    f = f.merge(beh, on=["a", "b"], how="left")
    assert len(f) == len(pool), "pairs join added rows"
    f["has_pair"] = f["lift_adj"].notna()
    f["log1p_k_m"] = np.log1p(f["k_m"].fillna(0))  # 0 when no pair (fewer than 5 sightings)
    f["log1p_n_m"] = np.log1p(f["a"].map(pairs.groupby("a")["n_m"].first()))  # NaN: A has no pair
    f = f.drop(columns="k_m")

    pos = pd.Series(np.arange(len(products)), index=products["product_id"].to_numpy())
    ia, ib = pos[f["a"]].to_numpy(), pos[f["b"]].to_numpy()
    cos = np.empty(len(f), dtype="float32")
    for s in range(0, len(f), 100_000):  # chunked to keep the sparse products small
        e = s + 100_000
        cos[s:e] = np.asarray(X[ia[s:e]].multiply(X[ib[s:e]]).sum(axis=1)).ravel()
    f["name_cosine"] = cos

    pop = counts.reindex(p.index, fill_value=0)
    aisle_rank = pop.groupby(p["aisle_id"]).rank(ascending=False, method="min")
    f["log1p_b_first_buys"] = np.log1p(f["b"].map(pop))
    f["b_aisle_pop_rank"] = f["b"].map(aisle_rank)

    f["same_aisle"] = f["a"].map(p["aisle_id"]).to_numpy() == f["b"].map(p["aisle_id"]).to_numpy()
    f["same_department"] = (f["a"].map(p["department_id"]).to_numpy()
                            == f["b"].map(p["department_id"]).to_numpy())
    f["b_reorder_rate"] = f["b"].map(product_reorder)  # NaN: B never bought in the window

    organic = p["product_name"].str.contains("organic", case=False).astype(bool)
    f["organic_a"] = f["a"].map(organic)
    f["organic_b"] = f["b"].map(organic)
    f["organic_switch"] = f["organic_a"] != f["organic_b"]
    return f


def event_rows(events, pair_feats, customer, truth):
    """Join pool features to events, add customer features and the aisle-truth label."""
    n_pool = events["a"].map(pair_feats.groupby("a").size()).fillna(0)
    rows = events[["event_id", "user_id", "order_id", "a"]].merge(pair_feats, on="a")
    assert len(rows) == n_pool.sum(), "event-pool join row count mismatch"

    rows = rows.join(customer, on="user_id")
    assert rows["cust_n_orders"].notna().all(), "customer missing from feature window"

    rows = rows.merge(truth.assign(in_truth=True), on=["order_id", "b"], how="left")
    assert len(rows) == n_pool.sum(), "truth join added rows"
    rows["label"] = (rows["in_truth"].fillna(False).astype(bool) & rows["same_aisle"]).astype("int8")

    rows[FEATURES] = rows[FEATURES].astype("float32")
    rows = rows[KEYS + FEATURES + ["label"]].sort_values(["event_id", "b"], kind="stable")
    return rows.reset_index(drop=True)


def side_lines(side):
    """Prior lines for this side: the training feature window, or all prior orders."""
    lines = load_prior_lines()[["order_id", "user_id", "product_id", "reordered", "order_number"]]
    if side == "train":
        lines, _ = split_last_prior(lines)
    return lines.drop(columns="order_number")


def build_side(side, products, X):
    """All feature rows for 'train' or 'eval'."""
    out_dir = WINDOW_DIR if side == "train" else EVAL_DIR
    events, truth, counts, pairs_path = side_inputs(side, products)
    pool = pd.read_parquet(out_dir / "pool.parquet")
    pairs = pd.read_parquet(pairs_path, columns=["a", "b", "lift_adj", "lift_low_adj", "k_m", "n_m"])

    lines = side_lines(side)
    if side == "train":
        orders = load_orders()[["order_id", "eval_set"]]
        assert_prior_only(lines["order_id"].unique(), orders)
        assert_prior_only(events["order_id"].unique(), orders)  # label orders are prior orders
        del orders
    product_reorder, customer = line_stats(lines)
    del lines
    gc.collect()

    feats = pair_features(pool, pairs, counts, products, X, product_reorder)
    rows = event_rows(events, feats, customer, truth)
    rows.to_parquet(out_dir / "features.parquet", index=False)
    return rows, events


def report(side, rows, events):
    """Shape, memory, label rate and NaN check for one side."""
    mem = rows.memory_usage(deep=True).sum() / 1e6
    pos_events = rows.groupby("event_id")["label"].max()
    print(f"\n=== {side} ===")
    print(f"shape {rows.shape[0]:,} x {rows.shape[1]}, memory {mem:,.0f} MB, "
          f"feature dtypes {sorted(set(str(t) for t in rows[FEATURES].dtypes))}")
    print(f"positive label rate: {rows['label'].mean():.3%} of rows; events with >= 1 positive: "
          f"{pos_events.sum():,} of {len(events):,} ({pos_events.sum() / len(events):.1%}, "
          f"should equal aisle pool recall)")
    nan = rows[FEATURES].isna().mean()
    bad = [c for c in FEATURES if c not in NAN_OK and nan[c] > 0]
    print("NaN share: " + ", ".join(f"{c} {nan[c]:.1%}" for c in FEATURES if nan[c] > 0))
    assert not bad, f"unexpected NaN in {bad}"
    assert (rows["has_pair"] == 0).eq(rows["lift_adj"].isna()).all(), "has_pair/lift mismatch"
    print("NaN check passed: NaN only in " + ", ".join(sorted(NAN_OK))
          + "; has_pair == 0 exactly where lift_adj is NaN")


def main():
    t0 = time.perf_counter()
    products = load_products()
    X = name_vectors(products)  # same vectoriser and settings as similarity_ranker

    with no_final_orders():
        train, train_events = build_side("train", products, X)
    report("train", train, train_events)
    print(f"[train] {time.perf_counter() - t0:,.0f}s, peak memory {peak_mb():,.0f} MB")
    del train
    gc.collect()

    evaluation, eval_events = build_side("eval", products, X)
    report("eval", evaluation, eval_events)
    print(f"[eval] {time.perf_counter() - t0:,.0f}s, peak memory {peak_mb():,.0f} MB")


if __name__ == "__main__":
    main()
