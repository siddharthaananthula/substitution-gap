"""Tests for the candidate pool in src/candidates.py."""
import pandas as pd

from src.candidates import SOURCES, build_pool
from src.rankers import popularity_ranker, similarity_ranker
from tests.test_rankers import COUNTS, PRODUCTS


def pool():
    pairs = pd.DataFrame({  # behaviour pairs: 1->4 crosses aisles, 1->2 overlaps the rankers
        "a": [1, 1, 4], "b": [2, 4, 5], "lift_low_adj": [3.0, 1.5, 2.0]})
    return build_pool([1, 3, 4], similarity_ranker(PRODUCTS),
                      popularity_ranker(PRODUCTS, COUNTS), pairs)


def test_pool_never_contains_a_and_has_no_duplicates():
    p = pool()
    assert not (p["a"] == p["b"]).any()
    assert not p.duplicated(["a", "b"]).any()


def test_pool_is_union_with_source_flags():
    p = pool().set_index(["a", "b"])
    # 1 -> 2 comes from all three sources but appears once
    assert p.loc[(1, 2), list(SOURCES)].tolist() == [True, True, True]
    # 1 -> 4 is behaviour only (other aisle), 1 -> 3 is aisle rankers only
    assert p.loc[(1, 4), list(SOURCES)].tolist() == [False, False, True]
    assert p.loc[(1, 3), "src_beh"] == False  # noqa: E712
    assert set(p.loc[1].index) == {2, 3, 4}
    # 4 is alone in its aisle: only its behaviour pair
    assert set(p.loc[4].index) == {5}


def test_pool_ignores_a_from_behaviour_pairs():
    pairs = pd.DataFrame({"a": [1], "b": [1], "lift_low_adj": [9.0]})
    p = build_pool([1], similarity_ranker(PRODUCTS), popularity_ranker(PRODUCTS, COUNTS), pairs)
    assert 1 not in set(p["b"])
