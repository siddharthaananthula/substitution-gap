"""Pair-level substitution lift.

For every regular product A and candidate product B, compare how often B is
bought for the first time when A is missing from an order, against when A is
bought. Lift = missing rate / bought rate. See DECISIONS.md D002, D012-D014.
"""
import numpy as np
import pandas as pd

MIN_ORDERS = 12    # D012: customer needs at least 12 prior orders
MIN_SHARE = 0.5    # D012: product in at least 50% of the customer's orders


def orders_per_user(lines):
    """Number of prior orders for each customer."""
    return lines.groupby("user_id")["order_id"].nunique()


def find_regulars(lines, n_orders, min_orders=MIN_ORDERS, min_share=MIN_SHARE):
    """One row per (customer, regular product)."""
    eligible = n_orders[n_orders >= min_orders]
    eligible_lines = lines[lines["user_id"].isin(eligible.index)]
    up = (
        eligible_lines.groupby(["user_id", "product_id"]).size()
        .rename("times_bought").reset_index()
    )
    up["share"] = up["times_bought"] / up["user_id"].map(eligible)
    return up.loc[up["share"] >= min_share, ["user_id", "product_id"]]


def _count(events, new_lines):
    """n: events per regular A. k: events per (A, B) where B was a first-time buy."""
    e = events[["order_id", "product_id"]].rename(columns={"product_id": "a"})
    n = e.groupby("a").size()
    k = e.merge(new_lines, on="order_id").groupby(["a", "b"]).size()
    return n, k


def count_batch(lines_b, reg_b):
    """Counts for one batch of customers: n_m, k_m, n_b, k_b."""
    orders_b = lines_b[["user_id", "order_id"]].drop_duplicates()
    expected = orders_b.merge(reg_b, on="user_id")

    bought = lines_b[["order_id", "product_id", "reordered"]]
    expected = expected.merge(bought, on=["order_id", "product_id"], how="left")

    missing = expected[expected["reordered"].isna()]
    present = expected[expected["reordered"] == 1]  # skip A's own first purchase

    new_lines = lines_b.loc[
        lines_b["reordered"] == 0, ["order_id", "product_id"]
    ].rename(columns={"product_id": "b"})

    n_m, k_m = _count(missing, new_lines)
    n_b, k_b = _count(present, new_lines)
    return n_m, k_m, n_b, k_b


def run_batches(lines, reg, n_batches=10, seed=42):
    """Split customers into batches, count each, and add the counts up."""
    users = reg["user_id"].unique()
    np.random.default_rng(seed).shuffle(users)

    parts = {"n_m": [], "k_m": [], "n_b": [], "k_b": []}
    for i, batch in enumerate(np.array_split(users, n_batches), start=1):
        lines_b = lines[lines["user_id"].isin(batch)]
        reg_b = reg[reg["user_id"].isin(batch)]
        for key, counts in zip(parts, count_batch(lines_b, reg_b)):
            parts[key].append(counts)
        print(f"batch {i}/{n_batches} done")

    # Add counts across batches (never average rates)
    return {
        key: pd.concat(vals).groupby(level=list(vals[0].index.names)).sum()
        for key, vals in parts.items()
    }


def compute_lift(totals, min_k=5, min_n=30, alpha=1):
    """Lift per (A, B), keeping only pairs with enough evidence."""
    pairs = totals["k_m"].rename("k_m").reset_index()
    pairs["n_m"] = pairs["a"].map(totals["n_m"])
    pairs["n_b"] = pairs["a"].map(totals["n_b"])
    pairs = pairs.merge(
        totals["k_b"].rename("k_b").reset_index(), on=["a", "b"], how="left"
    )
    pairs["k_b"] = pairs["k_b"].fillna(0)
    pairs = pairs.dropna(subset=["n_b"])
    pairs = pairs[(pairs["k_m"] >= min_k) & (pairs["n_m"] >= min_n)].copy()

    rate_m = pairs["k_m"] / pairs["n_m"]
    rate_b = (pairs["k_b"] + alpha) / (pairs["n_b"] + alpha)  # smoothing: no divide by zero
    pairs["lift"] = rate_m / rate_b
    return pairs.sort_values("lift", ascending=False).reset_index(drop=True)


def add_names(pairs, products):
    """Attach product names for A and B."""
    names = products.set_index("product_id")["product_name"]
    return pairs.assign(a_name=pairs["a"].map(names), b_name=pairs["b"].map(names))

def add_lift_lower(pairs, z=1.96, alpha=1):
    """Cautious lift: lower end of a 95% confidence interval. Rank by this."""
    k_b = pairs["k_b"] + alpha
    n_b = pairs["n_b"] + alpha
    se = np.sqrt(1 / pairs["k_m"] - 1 / pairs["n_m"] + 1 / k_b - 1 / n_b)
    pairs = pairs.assign(lift_low=np.exp(np.log(pairs["lift"]) - z * se))
    return pairs.sort_values("lift_low", ascending=False).reset_index(drop=True)

def adjust_for_exploration(pairs, totals):
    """Divide lift by the general 'try more new things' boost when a regular is missing."""
    rate_m = totals["k_m"].sum() / totals["n_m"].sum()
    rate_b = totals["k_b"].sum() / totals["n_b"].sum()
    ratio = rate_m / rate_b
    print("exploration ratio:", round(ratio, 3))
    return pairs.assign(
        lift_adj=pairs["lift"] / ratio,
        lift_low_adj=pairs["lift_low"] / ratio,
    )