"""Tests for src/features.py: the no-final-orders guard and feature/label logic (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import EVAL_DIR
from src.features import event_rows, line_stats, no_final_orders, pair_features
from src.load import DATA
from src.rankers import SUBSTITUTES, name_vectors
from tests.test_rankers import COUNTS, PRODUCTS


# --- no final orders in training ---------------------------------------------

@pytest.mark.parametrize("read, path", [
    (pd.read_csv, DATA / "order_products__train.csv"),  # final-order lines
    (pd.read_parquet, SUBSTITUTES),                      # pairs built from ALL prior orders
    (pd.read_parquet, EVAL_DIR / "events.parquet"),      # evaluation events
    (pd.read_parquet, EVAL_DIR / "pool.parquet"),
])
def test_guard_blocks_final_order_data(read, path):
    name = read.__name__
    with no_final_orders():
        with pytest.raises(RuntimeError, match="leakage"):
            getattr(pd, name)(path)  # looked up on pd, as src/load.py does


def test_guard_allows_other_files_and_restores(tmp_path):
    ok = tmp_path / "ok.csv"
    pd.DataFrame({"x": [1]}).to_csv(ok, index=False)
    real = pd.read_csv
    with no_final_orders():
        assert pd.read_csv(ok)["x"].tolist() == [1]
    assert pd.read_csv is real  # patched functions are restored


# --- features and label ----------------------------------------------------

@pytest.fixture
def feats():
    pool = pd.DataFrame({"a": [1, 1, 1], "b": [2, 3, 4]}, dtype="int32")
    pairs = pd.DataFrame({"a": [1], "b": [2], "lift_adj": [3.0], "lift_low_adj": [2.5],
                          "k_m": [9], "n_m": [40]})
    reorder = pd.Series({2: 0.5}, dtype="float32")  # 3 and 4 never bought
    return pair_features(pool, pairs, COUNTS, PRODUCTS, name_vectors(PRODUCTS), reorder)


def test_behaviour_features_nan_policy(feats):
    f = feats.set_index("b")
    assert f["has_pair"].tolist() == [True, False, False]
    assert f.loc[2, "lift_adj"] == 3.0 and f.loc[[3, 4], "lift_adj"].isna().all()
    assert f.loc[2, "log1p_k_m"] == pytest.approx(np.log1p(9))
    assert (f.loc[[3, 4], "log1p_k_m"] == 0).all()  # no pair -> 0 sightings
    assert (f["log1p_n_m"] == np.log1p(40)).all()   # A-level: same for every B


def test_structure_organic_and_similarity(feats):
    f = feats.set_index("b")
    assert f["same_aisle"].tolist() == [True, True, False]
    assert f["same_department"].tolist() == [True, True, False]
    assert f.loc[2, "organic_switch"] and not f.loc[3, "organic_switch"]  # Organic Whole Milk
    assert f.loc[2, "name_cosine"] > f.loc[3, "name_cosine"] > 0
    assert f.loc[2, "b_reorder_rate"] == 0.5 and f.loc[[3, 4], "b_reorder_rate"].isna().all()
    assert f.loc[3, "b_aisle_pop_rank"] == 2  # aisle 10 counts: 1 -> 50, 3 -> 30, 2 -> 10


def test_label_is_aisle_truth(feats):
    events = pd.DataFrame({"event_id": [0], "user_id": [7], "order_id": [10], "a": [1]},
                          dtype="int32")
    truth = pd.DataFrame({"order_id": [10, 10], "b": [2, 4]}, dtype="int32")
    customer = pd.DataFrame({"cust_first_buy_share": [0.2], "cust_n_orders": [15.0]},
                            index=pd.Index([7], name="user_id"), dtype="float32")
    rows = event_rows(events, feats, customer, truth).set_index("b")
    # 2: first-time buy in A's aisle -> 1; 4: first-time buy in another aisle -> 0; 3: not bought
    assert rows["label"].to_dict() == {2: 1, 3: 0, 4: 0}
    assert (rows["cust_n_orders"] == 15).all()


def test_line_stats():
    lines = pd.DataFrame({"order_id": [1, 1, 2, 3], "user_id": [5, 5, 5, 6],
                          "product_id": [1, 2, 1, 1], "reordered": [0, 0, 1, 0]})
    product, customer = line_stats(lines)
    assert product[1] == pytest.approx(1 / 3)
    assert customer.loc[5, "cust_first_buy_share"] == pytest.approx(2 / 3)
    assert customer.loc[5, "cust_n_orders"] == 2
