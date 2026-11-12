"""Evaluate substitute rankers on each customer's final (train) order. See DECISIONS.md D011.

An event is a regular product A (found from prior orders only) that is missing
from the customer's final order. Three truth sets per event:
  loose  = every first-time buy (reordered == 0) in that final order
  strict = first-time buys in the same department as A
  aisle  = first-time buys in the same aisle as A

Run with: python -m src.evaluate
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.load import load_prior_lines, load_products, load_train_lines
from src.pairs import find_regulars, orders_per_user
from src.rankers import (
    behaviour_dept_ranker,
    behaviour_ranker,
    first_buy_counts,
    hybrid_ranker,
    popularity_ranker,
    similarity_ranker,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
KS = (1, 3, 5)
GROUP_COL = {"strict": "department_id", "aisle": "aisle_id"}  # truth = same group as A
TRUTHS = ("loose", *GROUP_COL)
MAIN = "behaviour_dept"  # Task 5: the ranker every other ranker is compared against
TASK5_RANKERS = ("similarity", "popularity", "behaviour", MAIN)
TASK6_RANKERS = ("similarity", "popularity", MAIN, "hybrid_sim", "hybrid_pop")
HYBRID_VS = {"hybrid_sim": "similarity", "hybrid_pop": "popularity"}
N_BOOT = 200
SEED = 42


# --- events and truth (Task 3) ---------------------------------------------

def prior_regulars():
    """Regulars from prior lines only, exactly as in build_substitutes."""
    lines = load_prior_lines()
    reg = find_regulars(lines, orders_per_user(lines))
    del lines
    gc.collect()
    return reg


def build_events(reg, train_lines):
    """One row per (customer, final order, regular A) where A is missing from that order."""
    final = train_lines[["user_id", "order_id"]].drop_duplicates()
    assert final["user_id"].is_unique, "a customer has more than one train order"

    expected = reg.merge(final, on="user_id", how="inner")  # drops test customers
    assert len(expected) <= len(reg)

    bought = train_lines[["order_id", "product_id"]].assign(in_order=True)
    expected = expected.merge(bought, on=["order_id", "product_id"], how="left")
    assert len(expected) == len(reg.merge(final, on="user_id")), "join added rows"

    events = expected.loc[expected["in_order"].isna(), ["user_id", "order_id", "product_id"]]
    return events.rename(columns={"product_id": "a"}).reset_index(drop=True)


def first_time_buys(train_lines):
    """The truth set: (order_id, b) for every first-time buy in a final order."""
    new = train_lines.loc[train_lines["reordered"] == 0, ["order_id", "product_id"]]
    return new.rename(columns={"product_id": "b"}).reset_index(drop=True)


def summarise_events(events, truth, reg, train_lines, names):
    """Print the size and shape of the evaluation set."""
    eval_users = reg.loc[reg["user_id"].isin(train_lines["user_id"]), "user_id"].unique()
    per_customer = events.groupby("user_id").size()
    print(f"customers with regulars and a train order: {len(eval_users):,}")
    print(f"events (regular missing from final order): {len(events):,}")
    print(f"customers with at least one event: {events['user_id'].nunique():,}")
    print(f"median events per customer (with >= 1 event): {per_customer.median():.0f}")
    print(f"median events per customer (incl. 0): "
          f"{per_customer.reindex(eval_users, fill_value=0).median():.0f}")
    print(f"first-time buys in final orders: {len(truth):,}")
    top = events["a"].value_counts().head(10).rename("events").to_frame()
    top.insert(0, "product_name", top.index.map(names))
    print("top 10 regular products by events:")
    print(top.to_string())


# --- scoring (Task 5) --------------------------------------------------------

def same_group(a, b, group):
    """True where products a and b share a group (department or aisle)."""
    return a.map(group).to_numpy() == b.map(group).to_numpy()


def add_truth_sizes(events, truth, groups):
    """Number of truth products per event for every truth set (n_loose, n_strict, n_aisle)."""
    ev = events.copy()
    ev["n_loose"] = ev["order_id"].map(truth.groupby("order_id").size()).fillna(0)

    t = ev[["event_id", "order_id", "a"]].merge(truth, on="order_id")
    assert len(t) == ev["n_loose"].sum(), "truth join row count mismatch"
    for name, group in groups.items():
        same = same_group(t["a"], t["b"], group)
        ev[f"n_{name}"] = ev["event_id"].map(t.loc[same].groupby("event_id").size()).fillna(0)
    return ev.astype({f"n_{t}": "int16" for t in TRUTHS})


def candidate_table(rank, a_ids, k):
    """Long table (a, rank, b) of each ranker's top-k candidates for every A."""
    a_col, r_col, b_col = [], [], []
    for a in a_ids:
        top = rank(a, k).index.to_numpy()
        a_col.extend([a] * len(top))
        r_col.extend(range(1, len(top) + 1))
        b_col.extend(top)
    return pd.DataFrame({
        "a": np.array(a_col, dtype="int32"),
        "rank": np.array(r_col, dtype="int8"),
        "b": np.array(b_col, dtype="int32"),
    })


def score_events(events, cands, truth, groups, ks=KS):
    """Per-event coverage, hit@k and NDCG@k (binary relevance) for every truth set."""
    n_cands = events["a"].map(cands.groupby("a").size()).fillna(0)
    c = events[["event_id", "order_id", "a"]].merge(cands, on="a")
    assert len(c) == n_cands.sum(), "candidate join row count mismatch"

    c = c.merge(truth.assign(loose=True), on=["order_id", "b"], how="left")
    assert len(c) == n_cands.sum(), "truth join added rows"
    c["loose"] = c["loose"].fillna(False).astype(bool)
    for name, group in groups.items():
        c[name] = c["loose"] & same_group(c["a"], c["b"], group)
    c["disc"] = 1 / np.log2(c["rank"].astype(float) + 1)

    # ideal DCG for n relevant items at cut-off k: sum of the first min(n, k) discounts
    cum_disc = np.concatenate([[0.0], np.cumsum(1 / np.log2(np.arange(2, max(ks) + 2)))])

    out = pd.DataFrame({"event_id": events["event_id"],
                        "covered": events["event_id"].isin(c["event_id"]).to_numpy()})
    for k in ks:
        sub = c[c["rank"] <= k]
        for t in TRUTHS:
            g = sub.groupby("event_id")
            hit = g[t].any().reindex(events["event_id"], fill_value=False).to_numpy()
            dcg = (sub[t] * sub["disc"]).groupby(sub["event_id"]).sum()
            dcg = dcg.reindex(events["event_id"], fill_value=0.0).to_numpy()
            idcg = cum_disc[np.minimum(events[f"n_{t}"].to_numpy(), k)]
            out[f"hit_{t}_{k}"] = hit
            out[f"ndcg_{t}_{k}"] = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    return out


def bootstrap_weights(user_codes, n_boot=N_BOOT, seed=SEED):
    """n_boot x n_customers matrix: how many times each customer is drawn per resample."""
    n = user_codes.max() + 1
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])


def mean_diff_ci(diff, mask, user_codes, weights):
    """Mean of diff over events in mask, and a 95% CI from resampling customers."""
    n = weights.shape[1]
    num = np.bincount(user_codes[mask], weights=diff[mask], minlength=n)
    den = np.bincount(user_codes[mask], minlength=n).astype(float)
    boots = (weights @ num) / (weights @ den)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return num.sum() / den.sum(), lo, hi


def results_table(scores, names, user_codes, weights, ks=KS):
    """Task 5: one row per ranker x truth x k. diff_* = behaviour_dept minus this ranker."""
    main = scores[MAIN]
    rows = []
    for name in names:
        s = scores[name]
        common = s["covered"].to_numpy() & main["covered"].to_numpy()
        for t in TRUTHS:
            for k in ks:
                hit, ndcg = f"hit_{t}_{k}", f"ndcg_{t}_{k}"
                row = {
                    "ranker": name, "truth": t, "k": k,
                    "coverage": s["covered"].mean(),
                    "hit_all": s[hit].mean(), "ndcg_all": s[ndcg].mean(),
                    "n_common": int(common.sum()),
                    "hit_common": s.loc[common, hit].mean(),
                    "ndcg_common": s.loc[common, ndcg].mean(),
                }
                if name != MAIN:
                    diff = main[hit].to_numpy(float) - s[hit].to_numpy(float)
                    everywhere = np.ones(len(diff), dtype=bool)
                    for label, mask in (("all", everywhere), ("common", common)):
                        d, lo, hi = mean_diff_ci(diff, mask, user_codes, weights)
                        row.update({f"diff_{label}": d, f"diff_{label}_lo": lo,
                                    f"diff_{label}_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def hybrid_table(scores, names, user_codes, weights, ks=KS):
    """Task 6, all events: one row per ranker x truth x k; hybrids get hit@k diff vs fallback."""
    everywhere = np.ones(len(user_codes), dtype=bool)
    rows = []
    for name in names:
        s = scores[name]
        for t in TRUTHS:
            for k in ks:
                hit, ndcg = f"hit_{t}_{k}", f"ndcg_{t}_{k}"
                row = {"ranker": name, "truth": t, "k": k, "coverage": s["covered"].mean(),
                       "hit": s[hit].mean(), "ndcg": s[ndcg].mean()}
                if name in HYBRID_VS:
                    base = scores[HYBRID_VS[name]]
                    diff = s[hit].to_numpy(float) - base[hit].to_numpy(float)
                    d, lo, hi = mean_diff_ci(diff, everywhere, user_codes, weights)
                    row.update({"vs": HYBRID_VS[name], "diff": d, "diff_lo": lo, "diff_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def print_hybrid_table(table):
    """Compact view: hit@k and NDCG@k per ranker, then the hybrid CIs, in points."""
    pts = "{:.1f}".format
    for metric in ("hit", "ndcg"):
        wide = table.pivot(index=["truth", "k"], columns="ranker", values=metric)
        wide = (wide[list(TASK6_RANKERS)] * 100).reindex(TRUTHS, level=0)
        print(f"\n{metric}@k (%):")
        print(wide.to_string(float_format=pts))
    h = table.dropna(subset=["diff"]).copy()
    h["ci"] = h.apply(lambda r: f"{100 * r['diff']:+.1f} [{100 * r['diff_lo']:+.1f}, "
                                f"{100 * r['diff_hi']:+.1f}]", axis=1)
    h["comparison"] = h["ranker"] + " - " + h["vs"]
    wide = h.pivot(index=["truth", "k"], columns="comparison", values="ci").reindex(TRUTHS, level=0)
    print(f"\nhit@k difference, points [95% CI, {N_BOOT} customer resamples, seed {SEED}]:")
    print(wide.to_string())


EVAL_DIR = OUT / "eval_window"


def evaluation_set(products, cache=True):
    """Evaluation events (with event_id and truth sizes), truth and prior first-buy counts.

    Same steps and event order as main(), so event_ids match. Cached in data/processed/eval_window.
    """
    paths = {n: EVAL_DIR / f"{n}.parquet" for n in ("events", "truth", "first_buy_counts")}
    if cache and all(p.exists() for p in paths.values()):
        counts = pd.read_parquet(paths["first_buy_counts"]).set_index("product_id")["count"]
        return pd.read_parquet(paths["events"]), pd.read_parquet(paths["truth"]), counts

    reg = prior_regulars()
    counts = first_buy_counts()
    train_lines = load_train_lines()[["order_id", "user_id", "product_id", "reordered"]]
    events = build_events(reg, train_lines)
    truth = first_time_buys(train_lines)
    del reg, train_lines
    gc.collect()
    events["event_id"] = np.arange(len(events), dtype="int32")
    groups = {t: products.set_index("product_id")[c] for t, c in GROUP_COL.items()}
    events = add_truth_sizes(events, truth, groups)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(paths["events"], index=False)
    truth.to_parquet(paths["truth"], index=False)
    counts.rename("count").rename_axis("product_id").reset_index().to_parquet(
        paths["first_buy_counts"], index=False)
    return events, truth, counts


def ceilings(events, covered):
    """Best possible hit rate: share of events with at least one truth product."""
    rows = []
    for t in TRUTHS:
        has = events[f"n_{t}"].to_numpy() > 0
        rows.append({"truth": t, "all_events": has.mean(),
                     f"{MAIN}_covered": has[covered].mean()})
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    reg = prior_regulars()
    counts = first_buy_counts()  # prior orders only
    gc.collect()

    train_lines = load_train_lines()[["order_id", "user_id", "product_id", "reordered"]]
    products = load_products()
    names = products.set_index("product_id")["product_name"]
    groups = {t: products.set_index("product_id")[col] for t, col in GROUP_COL.items()}

    events = build_events(reg, train_lines)
    truth = first_time_buys(train_lines)
    assert not truth.duplicated().any(), "a product appears twice in one order"
    summarise_events(events, truth, reg, train_lines, names)
    del reg, train_lines
    gc.collect()

    events["event_id"] = np.arange(len(events), dtype="int32")
    events = add_truth_sizes(events, truth, groups)

    similarity = similarity_ranker(products)
    popularity = popularity_ranker(products, counts)
    behaviour_dept = behaviour_dept_ranker(products)
    rankers = {
        "similarity": similarity,
        "popularity": popularity,
        "behaviour": behaviour_ranker(),
        MAIN: behaviour_dept,
        "hybrid_sim": hybrid_ranker(behaviour_dept, similarity),
        "hybrid_pop": hybrid_ranker(behaviour_dept, popularity),
    }
    a_ids = events["a"].unique()
    print(f"\ndistinct regular products in events: {len(a_ids):,}")
    scores = {}
    for name, rank in rankers.items():
        cands = candidate_table(rank, a_ids, max(KS))
        assert not cands.duplicated(["a", "b"]).any(), f"{name} repeats a candidate"
        assert not (cands["a"] == cands["b"]).any(), f"{name} returned A"
        scores[name] = score_events(events, cands, truth, groups)
        print(f"  scored {name}: {len(cands):,} candidate rows")

    user_codes = pd.factorize(events["user_id"])[0]
    weights = bootstrap_weights(user_codes)

    ceil = ceilings(events, scores[MAIN]["covered"].to_numpy())
    print("\nCeiling (share of events with >= 1 truth product):")
    print(ceil.to_string(index=False, float_format="{:.1%}".format))

    OUT.mkdir(parents=True, exist_ok=True)
    results_table(scores, TASK5_RANKERS, user_codes, weights).to_csv(
        OUT / "eval_results.csv", index=False)
    hybrid = hybrid_table(scores, TASK6_RANKERS, user_codes, weights)
    hybrid.to_csv(OUT / "eval_hybrid.csv", index=False)

    print("\ncoverage, all events: " + ", ".join(
        f"{n} {scores[n]['covered'].mean():.1%}" for n in TASK6_RANKERS))
    print_hybrid_table(hybrid)
    print(f"\nsaved {OUT / 'eval_results.csv'} (Task 5 table) and "
          f"{OUT / 'eval_hybrid.csv'}  ({time.perf_counter() - t0:,.0f}s)")


if __name__ == "__main__":
    main()
