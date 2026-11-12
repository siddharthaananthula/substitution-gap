"""Acceptance model: LightGBM lambdarank over the candidate pool, compared with hybrid_pop.

Train: training events (label = last prior order) with at least one positive;
       early stopping on a 20% customer split of those training events.
Test:  all 65,476 evaluation events (final orders), never used for training or tuning.

Run with: python -m src.model
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.build_substitutes import OUT, peak_mb
from src.evaluate import (
    EVAL_DIR,
    GROUP_COL,
    KS,
    N_BOOT,
    SEED,
    TRUTHS,
    bootstrap_weights,
    candidate_table,
    evaluation_set,
    mean_diff_ci,
    score_events,
)
from src.features import BEHAVIOUR, FEATURES
from src.load import load_products
from src.rankers import behaviour_dept_ranker, hybrid_ranker, popularity_ranker
from src.train_split import WINDOW_DIR

PARAMS = dict(objective="lambdarank", num_leaves=31, learning_rate=0.05, n_estimators=500,
              min_child_samples=20, random_state=SEED, verbose=-1)
EARLY_STOP = 50
VALID_SHARE = 0.2
BASELINE = "hybrid_pop"
EXAMPLES = ("Banana", "Organic Whole Milk", "Organic Hass Avocado")


def positive_events_only(rows):
    """Keep training events with at least one positive label (all-zero groups give no gradient)."""
    keep = rows.groupby("event_id")["label"].transform("max") == 1
    return rows[keep.to_numpy()].reset_index(drop=True)


def customer_split(rows, share=VALID_SHARE, seed=SEED):
    """Boolean mask: True for rows of customers held out for early stopping."""
    users = np.sort(rows["user_id"].unique())
    rng = np.random.default_rng(seed)
    valid_users = rng.choice(users, size=int(round(share * len(users))), replace=False)
    return rows["user_id"].isin(valid_users).to_numpy()


def group_sizes(rows):
    """Rows per event, in row order (rows must be sorted by event_id)."""
    assert rows["event_id"].is_monotonic_increasing, "rows must be sorted by event_id"
    return rows.groupby("event_id", sort=True).size().to_numpy()


def train_ranker(train_rows, features):
    """Fit LGBMRanker with early stopping on held-out training customers."""
    valid = customer_split(train_rows)
    tr, va = train_rows[~valid], train_rows[valid]
    assert not set(tr["user_id"]) & set(va["user_id"]), "customer in both fit and valid"
    model = lgb.LGBMRanker(**PARAMS)
    model.fit(
        tr[features], tr["label"], group=group_sizes(tr),
        eval_X=(va[features],), eval_y=(va["label"],), eval_group=[group_sizes(va)], eval_at=[5],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    print(f"  fit: {tr['event_id'].nunique():,} events / {len(tr):,} rows; "
          f"valid: {va['event_id'].nunique():,} events / {len(va):,} rows "
          f"({va['user_id'].nunique():,} customers)")
    print(f"  best iteration {model.best_iteration_}, "
          f"valid ndcg@5 {model.best_score_['valid_0']['ndcg@5']:.4f}")
    return model


def model_candidates(rows, scores, k=max(KS)):
    """Per-event top-k (event_id, rank, b) from model scores; ties broken by product_id."""
    s = rows[["event_id", "b"]].assign(score=scores)
    s = s.sort_values(["event_id", "score", "b"], ascending=[True, False, True], kind="stable")
    s["rank"] = s.groupby("event_id").cumcount() + 1
    top = s[s["rank"] <= k]
    return top.astype({"rank": "int8"})[["event_id", "rank", "b", "score"]]


def results(scores, user_codes, weights, ks=KS):
    """One row per ranker x truth x k: coverage, hit, ndcg, and hit diff vs hybrid_pop with CI."""
    base = scores[BASELINE]
    everywhere = np.ones(len(user_codes), dtype=bool)
    rows = []
    for name, s in scores.items():
        for t in TRUTHS:
            for k in ks:
                hit, ndcg = f"hit_{t}_{k}", f"ndcg_{t}_{k}"
                row = {"ranker": name, "truth": t, "k": k, "coverage": s["covered"].mean(),
                       "hit": s[hit].mean(), "ndcg": s[ndcg].mean()}
                if name != BASELINE:
                    diff = s[hit].to_numpy(float) - base[hit].to_numpy(float)
                    d, lo, hi = mean_diff_ci(diff, everywhere, user_codes, weights)
                    row.update({"diff": d, "diff_lo": lo, "diff_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def print_results(table, order):
    """Compact view in percentage points."""
    for metric in ("hit", "ndcg"):
        wide = table.pivot(index=["truth", "k"], columns="ranker", values=metric)
        print(f"\n{metric}@k (%), all evaluation events:")
        print((wide[list(order)] * 100).reindex(TRUTHS, level=0).to_string(float_format="{:.2f}".format))
    d = table.dropna(subset=["diff"]).copy()
    d["ci"] = d.apply(lambda r: f"{100 * r['diff']:+.2f} [{100 * r['diff_lo']:+.2f}, "
                                f"{100 * r['diff_hi']:+.2f}]", axis=1)
    d["ranker"] = d["ranker"] + f" - {BASELINE}"
    wide = d.pivot(index=["truth", "k"], columns="ranker", values="ci").reindex(TRUTHS, level=0)
    print(f"\nhit@k difference vs {BASELINE}, points [95% CI, {N_BOOT} customer resamples, seed {SEED}]:")
    print(wide.to_string())


def print_examples(eval_rows, model_cands, events, names, truth, aisle):
    """Top-5 model candidates for one evaluation event of each example product."""
    for name in EXAMPLES:
        a = int(names.index[names == name][0])
        ev = events[events["a"] == a].iloc[0]
        top = model_cands[model_cands["event_id"] == ev["event_id"]]
        hits = set(truth.loc[truth["order_id"] == ev["order_id"], "b"])
        feats = eval_rows[eval_rows["event_id"] == ev["event_id"]].set_index("b")
        print(f"\n=== {name}: event {ev['event_id']}, customer {ev['user_id']} ===")
        for _, r in top.iterrows():
            b = int(r["b"])
            tag = ("  <- first-time buy, same aisle" if b in hits and aisle[b] == aisle[a]
                   else "  <- first-time buy" if b in hits else "")
            print(f"  {int(r['rank'])}. {r['score']:+.3f}  {names[b]:<40} lift_low_adj "
                  f"{feats.loc[b, 'lift_low_adj']:.2f}  pop rank {feats.loc[b, 'b_aisle_pop_rank']:.0f}{tag}")
        in_aisle = sorted(names[b] for b in hits if aisle[b] == aisle[a])
        print(f"  actual first-time buys in A's aisle: {in_aisle or 'none'}")


def main():
    t0 = time.perf_counter()
    products = load_products()
    names = products.set_index("product_id")["product_name"]
    aisle = products.set_index("product_id")["aisle_id"]
    groups = {t: products.set_index("product_id")[c] for t, c in GROUP_COL.items()}

    # --- training (training feature window only) ---
    train_all = pd.read_parquet(WINDOW_DIR / "features.parquet")
    train_rows = positive_events_only(train_all)
    print(f"training events: {train_all['event_id'].nunique():,} -> "
          f"{train_rows['event_id'].nunique():,} with >= 1 positive; rows "
          f"{len(train_all):,} -> {len(train_rows):,}; customers {train_rows['user_id'].nunique():,}; "
          f"positive rate {train_rows['label'].mean():.2%}")
    del train_all

    no_beh = [f for f in FEATURES if f not in BEHAVIOUR]
    print("\nmodel (all features):")
    model = train_ranker(train_rows, FEATURES)
    print("\nmodel_no_behaviour (without " + ", ".join(BEHAVIOUR) + "):")
    model_nb = train_ranker(train_rows, no_beh)
    model.booster_.save_model(str(OUT / "model.txt"))
    model_nb.booster_.save_model(str(OUT / "model_no_behaviour.txt"))

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15)
    print("\nfeature importance (gain share), model:")
    print(imp.to_string(float_format="{:.1%}".format))

    # --- evaluation (all prior orders for features; final orders for truth) ---
    events, truth, counts = evaluation_set(products)
    eval_rows = pd.read_parquet(EVAL_DIR / "features.parquet")
    assert set(eval_rows["event_id"]) <= set(events["event_id"])

    cands = {}
    for name, m, feats in (("model", model, FEATURES), ("model_no_behaviour", model_nb, no_beh)):
        cands[name] = model_candidates(eval_rows, m.predict(eval_rows[feats]))
    hybrid = hybrid_ranker(behaviour_dept_ranker(products), popularity_ranker(products, counts))
    a_ids = events["a"].unique()
    scores = {name: score_events(events, c[["event_id", "rank", "b"]], truth, groups)
              for name, c in cands.items()}
    scores[BASELINE] = score_events(events, candidate_table(hybrid, a_ids, max(KS)), truth, groups)

    user_codes = pd.factorize(events["user_id"])[0]
    weights = bootstrap_weights(user_codes)
    table = results(scores, user_codes, weights)
    table.to_csv(OUT / "eval_model.csv", index=False)

    print("\ncoverage, all events: " + ", ".join(
        f"{n} {s['covered'].mean():.1%}" for n, s in scores.items()))
    print_results(table, (BASELINE, "model", "model_no_behaviour"))
    print_examples(eval_rows, cands["model"], events, names, truth, aisle)
    print(f"\nsaved {OUT / 'model.txt'} and {OUT / 'eval_model.csv'}  "
          f"({time.perf_counter() - t0:,.0f}s, peak memory {peak_mb():,.0f} MB)")


if __name__ == "__main__":
    main()
