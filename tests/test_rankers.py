"""Tests for src/rankers.py on a tiny synthetic catalogue."""
import pandas as pd
import pytest

from src.rankers import behaviour_ranker, popularity_ranker, similarity_ranker

PRODUCTS = pd.DataFrame({
    "product_id": pd.array([1, 2, 3, 4, 5], dtype="int32"),
    "product_name": pd.array(["Whole Milk", "Organic Whole Milk", "Skim Milk",
                              "Whole Milk Yogurt", "Banana"], dtype="string"),
    "aisle_id": pd.array([10, 10, 10, 20, 30], dtype="int16"),
})
COUNTS = pd.Series({1: 50, 2: 10, 3: 30, 4: 99, 5: 99})


@pytest.fixture
def behaviour(tmp_path):
    path = tmp_path / "subs.parquet"
    pd.DataFrame({
        "a": pd.array([1, 1, 1, 1], dtype="int32"),
        "b": pd.array([2, 3, 4, 1], dtype="int32"),
        "lift_low_adj": [5.0, 1.5, 3.0, 9.0],  # 3 below cut-off; (1, 1) is A itself
    }).to_parquet(path)
    return behaviour_ranker(path)


@pytest.fixture
def rankers(behaviour):
    return {
        "similarity": similarity_ranker(PRODUCTS),
        "behaviour": behaviour,
        "popularity": popularity_ranker(PRODUCTS, COUNTS),
    }


@pytest.mark.parametrize("name", ["similarity", "behaviour", "popularity"])
def test_never_returns_a(rankers, name):
    for a in PRODUCTS["product_id"]:
        assert a not in rankers[name](a, 10).index


@pytest.mark.parametrize("name", ["similarity", "popularity"])
def test_aisle_rankers_stay_in_aisle(rankers, name):
    # Product 4 (aisle 20) shares words with milk but is in another aisle
    assert set(rankers[name](1, 10).index) == {2, 3}
    assert rankers[name](5, 10).empty  # Banana is alone in its aisle


def test_scores_sorted_and_k_respected(rankers):
    top = rankers["popularity"](1, 1)
    assert list(top.index) == [3]  # 30 first-time buys beats 10
    assert rankers["similarity"](1, 10).index[0] == 2  # "Organic Whole Milk" closest


def test_behaviour_uses_cutoff(behaviour):
    assert list(behaviour(1, 10).index) == [2, 4]  # 5.0 > 3.0; 3 dropped (1.5 <= 2); A dropped
    assert behaviour(2, 10).empty  # no pairs for this A


def test_unknown_product_returns_empty(rankers):
    for rank in rankers.values():
        assert rank(12345, 5).empty
