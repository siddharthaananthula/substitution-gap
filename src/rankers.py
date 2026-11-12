"""Three rankers that suggest substitute products B for a missing product A.

Each builder returns a function rank(a, k) -> pd.Series of up to k candidates,
indexed by product_id B, values = score, best first. A itself is never returned.
An empty Series means the ranker has no candidate for A.

Run with: python -m src.rankers  (prints top-5 examples)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.load import load_order_products, load_products

SUBSTITUTES = Path(__file__).resolve().parents[1] / "data" / "processed" / "substitutes.parquet"
MIN_LIFT_LOW_ADJ = 2  # D015: confident substitute set


def _top_k(ids, scores, a, k):
    """Drop A and zero scores, sort by score (ties by product_id), keep k."""
    s = pd.Series(scores, index=pd.Index(ids, name="b"), name="score")
    s = s[(s.index != a) & (s > 0)]
    order = np.lexsort((s.index.to_numpy(), -s.to_numpy()))
    return s.iloc[order[:k]]


def _aisle_members(products):
    """aisle_id -> array of product positions, and product_id -> position."""
    pos = pd.Series(np.arange(len(products)), index=products["product_id"].to_numpy())
    members = products.groupby("aisle_id").indices
    return pos, members


def similarity_ranker(products, analyzer="char_wb", ngram_range=(3, 4)):
    """Same-aisle candidates ranked by TF-IDF cosine similarity of product names."""
    products = products[["product_id", "product_name", "aisle_id"]].reset_index(drop=True)
    tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=True)
    X = tfidf.fit_transform(products["product_name"].astype(str))  # rows are L2-normalised
    ids = products["product_id"].to_numpy()
    aisles = products["aisle_id"].to_numpy()
    pos, members = _aisle_members(products)

    @lru_cache(maxsize=None)  # caches only the top-k result, not whole-aisle scores
    def _rank(a, k):
        if a not in pos.index:
            return _top_k([], [], a, k)
        i = pos[a]
        cand = members[aisles[i]]
        sims = (X[cand] @ X[i].T).toarray().ravel()  # cosine = dot of unit vectors
        return _top_k(ids[cand], sims, a, k)

    def rank(a, k):
        return _rank(int(a), int(k))

    return rank


def _load_confident_pairs(path, min_score):
    """Behavioural pairs with lift_low_adj > min_score, excluding A == B."""
    pairs = pd.read_parquet(path, columns=["a", "b", "lift_low_adj"])
    return pairs[(pairs["lift_low_adj"] > min_score) & (pairs["a"] != pairs["b"])]


def _pairs_ranker(pairs):
    """rank(a, k) over a pairs table, ranked by lift_low_adj."""
    by_a = {a: (g["b"].to_numpy(), g["lift_low_adj"].to_numpy())
            for a, g in pairs.groupby("a")}

    def rank(a, k):
        ids, scores = by_a.get(int(a), ([], []))
        return _top_k(ids, scores, a, k)

    return rank


def behaviour_ranker(path=SUBSTITUTES, min_score=MIN_LIFT_LOW_ADJ):
    """Candidates from behavioural pairs with lift_low_adj > min_score, ranked by it."""
    return _pairs_ranker(_load_confident_pairs(path, min_score))


def behaviour_dept_ranker(products, path=SUBSTITUTES, min_score=MIN_LIFT_LOW_ADJ):
    """behaviour_ranker, keeping only candidates in the same department as A."""
    pairs = _load_confident_pairs(path, min_score)
    dept = products.set_index("product_id")["department_id"]
    n_before = len(pairs)
    pairs = pairs[pairs["a"].map(dept).to_numpy() == pairs["b"].map(dept).to_numpy()]
    assert len(pairs) <= n_before
    return _pairs_ranker(pairs)


def first_buy_counts():
    """How often each product was a first-time buy (reordered == 0) in PRIOR orders."""
    lines = load_order_products()[["product_id", "reordered"]]  # prior file only
    return lines.loc[lines["reordered"] == 0, "product_id"].value_counts()


def popularity_ranker(products, counts):
    """Same-aisle candidates ranked by how often they were a first-time buy in prior orders."""
    products = products[["product_id", "aisle_id"]].reset_index(drop=True)
    ids = products["product_id"].to_numpy()
    aisles = products["aisle_id"].to_numpy()
    pop = counts.reindex(ids, fill_value=0).to_numpy()
    pos, members = _aisle_members(products)

    def rank(a, k):
        a = int(a)
        if a not in pos.index:
            return _top_k([], [], a, k)
        cand = members[aisles[pos[a]]]
        return _top_k(ids[cand], pop[cand], a, k)

    return rank


def main():
    products = load_products()
    names = products.set_index("product_id")["product_name"]
    rankers = {
        "similarity": similarity_ranker(products),
        "behaviour": behaviour_ranker(),
        "behaviour_dept": behaviour_dept_ranker(products),
        "popularity": popularity_ranker(products, first_buy_counts()),
    }
    for name in ("Banana", "Organic Whole Milk", "Organic Hass Avocado"):
        match = products.loc[products["product_name"] == name, "product_id"]
        assert len(match) == 1, f"{name!r} matches {len(match)} products"
        a = int(match.iloc[0])
        print(f"\n=== {name} (product {a}) ===")
        for rname, rank in rankers.items():
            top = rank(a, 5)
            print(f"  {rname}:")
            if top.empty:
                print("    (no candidates)")
            for b, score in top.items():
                print(f"    {score:>10.3f}  {names[b]}")


if __name__ == "__main__":
    main()
