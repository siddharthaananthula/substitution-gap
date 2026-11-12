"""Candidate pool for the acceptance model.

For each regular product A, the pool is the union of:
  top 20 similarity_ranker, top 20 popularity_ranker, and every behaviour pair
  with lift_low_adj > 1 (any department). Never A, no duplicates.
The pool depends only on A, so it is built once per A and joined to events.
Training uses the feature window's data, evaluation uses all prior orders.

Run with: python -m src.candidates
"""
import time

import pandas as pd

from src.build_substitutes import peak_mb
from src.evaluate import GROUP_COL, TRUTHS, candidate_table, evaluation_set, same_group, EVAL_DIR
from src.load import load_products
from src.rankers import SUBSTITUTES, _load_confident_pairs, popularity_ranker, similarity_ranker
from src.train_split import WINDOW_DIR

POOL_K = 20          # top-k taken from each aisle ranker
POOL_MIN_LIFT = 1    # behaviour pairs kept in the pool: lift_low_adj > 1
SOURCES = ("src_sim", "src_pop", "src_beh")
CEILING_D016 = {"loose": 0.786, "strict": 0.295, "aisle": 0.113}


def build_pool(a_ids, similarity, popularity, pairs, k=POOL_K):
    """One row per (a, b) in the pool, with a True/False flag for each source."""
    sim = candidate_table(similarity, a_ids, k)[["a", "b"]].assign(src_sim=True)
    pop = candidate_table(popularity, a_ids, k)[["a", "b"]].assign(src_pop=True)
    beh = pairs.loc[pairs["a"].isin(a_ids), ["a", "b"]].astype("int32").assign(src_beh=True)
    for name, part in (("similarity", sim), ("popularity", pop), ("behaviour", beh)):
        assert not part.duplicated(["a", "b"]).any(), f"{name} repeats a candidate"

    pool = sim.merge(pop, on=["a", "b"], how="outer").merge(beh, on=["a", "b"], how="outer")
    assert len(pool) <= len(sim) + len(pop) + len(beh)
    pool[list(SOURCES)] = pool[list(SOURCES)].fillna(False).astype(bool)
    pool = pool[pool["a"] != pool["b"]].sort_values(["a", "b"]).reset_index(drop=True)
    assert not pool.duplicated(["a", "b"]).any()
    return pool.astype({"a": "int32", "b": "int32"})


def side_inputs(side, products):
    """Events, truth, popularity counts and pairs path for 'train' or 'eval'."""
    if side == "train":
        events = pd.read_parquet(WINDOW_DIR / "events.parquet")
        truth = pd.read_parquet(WINDOW_DIR / "truth.parquet")
        counts = pd.read_parquet(WINDOW_DIR / "first_buy_counts.parquet")
        return events, truth, counts.set_index("product_id")["count"], WINDOW_DIR / "substitutes.parquet"
    events, truth, counts = evaluation_set(products)
    return events, truth, counts, SUBSTITUTES


def pool_recall(events, pool, truth, groups):
    """Per event: is at least one truth product in the pool (overall and per source)?"""
    n_pool = events["a"].map(pool.groupby("a").size()).fillna(0)
    c = events[["event_id", "order_id", "a"]].merge(pool, on="a")
    assert len(c) == n_pool.sum(), "pool join row count mismatch"
    c = c.merge(truth.assign(loose=True), on=["order_id", "b"], how="left")
    assert len(c) == n_pool.sum(), "truth join added rows"
    c["loose"] = c["loose"].fillna(False).astype(bool)
    for name, group in groups.items():
        c[name] = c["loose"] & same_group(c["a"], c["b"], group)

    out = pd.DataFrame({"event_id": events["event_id"].to_numpy(), "pool_size": n_pool.to_numpy()})
    for t in TRUTHS:
        hits = c[c[t]]
        out[f"recall_{t}"] = out["event_id"].isin(hits["event_id"]).to_numpy()
        for s in SOURCES:
            out[f"recall_{t}_{s}"] = out["event_id"].isin(hits.loc[hits[s], "event_id"]).to_numpy()
    return out


def main():
    t0 = time.perf_counter()
    products = load_products()
    groups = {t: products.set_index("product_id")[c] for t, c in GROUP_COL.items()}
    similarity = similarity_ranker(products)  # catalogue only: same for both sides

    tables = []
    for side, out_dir in (("train", WINDOW_DIR), ("eval", EVAL_DIR)):
        events, truth, counts, pairs_path = side_inputs(side, products)
        pairs = _load_confident_pairs(pairs_path, POOL_MIN_LIFT)
        pool = build_pool(events["a"].unique(), similarity,
                          popularity_ranker(products, counts), pairs)
        pool.to_parquet(out_dir / "pool.parquet", index=False)

        rec = pool_recall(events, pool, truth, groups)
        print(f"\n=== {side}: {len(events):,} events, {events['a'].nunique():,} distinct A, "
              f"{len(pool):,} pool rows ===")
        print(f"pool size per event: mean {rec['pool_size'].mean():.1f}, "
              f"median {rec['pool_size'].median():.0f}, min {rec['pool_size'].min():.0f}")
        per_event = events[["a"]].merge(pool, on="a")
        print("pool rows per event from each source: " + ", ".join(
            f"{s[4:]} {per_event[s].sum() / len(events):.1f}" for s in SOURCES))

        for t in TRUTHS:
            row = {"side": side, "truth": t,
                   "ceiling": (events[f"n_{t}"] > 0).mean(),
                   "recall": rec[f"recall_{t}"].mean()}
            row["recall/ceiling"] = row["recall"] / row["ceiling"]
            row.update({f"via {s[4:]}": rec[f"recall_{t}_{s}"].mean() for s in SOURCES})
            if side == "eval":
                row["ceiling D016"] = CEILING_D016[t]
            tables.append(row)
        print(f"[{side}] done, {time.perf_counter() - t0:,.0f}s, peak memory {peak_mb():,.0f} MB")

    table = pd.DataFrame(tables)
    pct = [c for c in table.columns if c not in ("side", "truth")]
    show = table.copy()
    show[pct] = show[pct].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    print("\nCandidate recall (share of events with >= 1 truth product in the pool):")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
