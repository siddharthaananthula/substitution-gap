"""Tests for src/rankers.py on a tiny synthetic catalogue."""
import pandas as pd
import pytest

from src.rankers import (
    behaviour_dept_ranker,
    behaviour_ranker,
    hybrid_ranker,
    popularity_ranker,
    similarity_ranker,
)

PRODUCTS = pd.DataFrame({
    "product_id": pd.array([1, 2, 3, 4, 5], dtype="int32"),
    "product_name": pd.array(["Whole Milk", "Organic Whole Milk", "Skim Milk",
                              "Whole Milk Yogurt", "Banana"], dtype="string"),
    "aisle_id": pd.array([10, 10, 10, 20, 30], dtype="int16"),
    "department_id": pd.array([1, 1, 1, 2, 3], dtype="int8"),
})
COUNTS = pd.Series({1: 50, 2: 10, 3: 30, 4: 99, 5: 99})


@pytest.fixture
def pairs_path(tmp_path):
    path = tmp_path / "subs.parquet"
    pd.DataFrame({
        "a": pd.array([1, 1, 1, 1, 4], dtype="int32"),
        "b": pd.array([2, 3, 4, 1, 5], dtype="int32"),
        # 3 below cut-off; (1, 1) is A itself; 4 and 5 are other departments
        "lift_low_adj": [5.0, 1.5, 3.0, 9.0, 4.0],
    }).to_parquet(path)
    return path


@pytest.fixture
def behaviour(pairs_path):
    return behaviour_ranker(pairs_path)


@pytest.fixture
def behaviour_dept(pairs_path):
    return behaviour_dept_ranker(PRODUCTS, pairs_path)


@pytest.fixture
def rankers(behaviour, behaviour_dept):
    return {
        "similarity": similarity_ranker(PRODUCTS),
        "behaviour": behaviour,
        "behaviour_dept": behaviour_dept,
        "popularity": popularity_ranker(PRODUCTS, COUNTS),
        "hybrid_pop": hybrid_ranker(behaviour_dept, popularity_ranker(PRODUCTS, COUNTS)),
        "hybrid_sim": hybrid_ranker(behaviour_dept, similarity_ranker(PRODUCTS)),
    }


@pytest.mark.parametrize("name", ["similarity", "behaviour", "behaviour_dept", "popularity",
                                  "hybrid_pop", "hybrid_sim"])
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


def test_behaviour_dept_stays_in_department(behaviour, behaviour_dept):
    dept = PRODUCTS.set_index("product_id")["department_id"]
    assert list(behaviour(1, 10).index) == [2, 4]  # original keeps cross-department 4
    assert list(behaviour_dept(1, 10).index) == [2]  # 4 is in another department
    assert behaviour_dept(4, 10).empty  # its only pair (4 -> 5) crosses departments
    for a in PRODUCTS["product_id"]:
        for b in behaviour_dept(a, 10).index:
            assert dept[b] == dept[a]


def test_hybrid_puts_primary_first_then_fills(rankers):
    # behaviour_dept(1) = [2]; popularity in aisle 10 for A=1 = [3 (30), 2 (10)]
    top = rankers["hybrid_pop"](1, 5)
    assert list(top.index) == [2, 3]  # 2 from primary, 3 fills; 2 not repeated
    assert top.index.is_unique
    assert list(rankers["hybrid_pop"](1, 1).index) == [2]  # k = 1: primary only


def test_hybrid_uses_fallback_when_primary_empty(rankers):
    # A=3 has no behavioural pairs, so hybrid_pop equals popularity
    assert list(rankers["hybrid_pop"](3, 5).index) == list(rankers["popularity"](3, 5).index)


def test_hybrid_fills_despite_overlap():
    def primary(a, k):
        return pd.Series([9.0], index=pd.Index([2], name="b")).iloc[:k]

    def fallback(a, k):  # overlaps with primary in its first slot
        return pd.Series([5.0, 4.0, 3.0], index=pd.Index([2, 3, 4], name="b")).iloc[:k]

    assert list(hybrid_ranker(primary, fallback)(1, 3).index) == [2, 3, 4]


def test_unknown_product_returns_empty(rankers):
    for rank in rankers.values():
        assert rank(12345, 5).empty
