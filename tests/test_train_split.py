"""Tests for the time-based training split in src/train_split.py."""
import pandas as pd
import pytest

from src.train_split import assert_prior_only, split_last_prior
from tests.test_pairs import FILLER, make_lines


@pytest.fixture
def lines():
    # customers with different numbers of orders; product 7 only in each last order
    return make_lines({
        1: [[FILLER, 1], [FILLER, 2], [FILLER, 7]],
        2: [[FILLER], [FILLER, 1], [FILLER, 3], [FILLER, 4], [FILLER, 7]],
        3: [[FILLER, 1, 2], [FILLER, 7]],
    })


def test_label_is_last_prior_order(lines):
    _, label = split_last_prior(lines)
    last = lines.groupby("user_id")["order_number"].max()
    got = label.groupby("user_id")["order_number"].agg(["min", "max"])
    assert (got["min"] == last).all() and (got["max"] == last).all()
    assert label.groupby("user_id")["order_id"].nunique().eq(1).all()  # one order each
    assert set(label["product_id"]) == {FILLER, 7}


def test_window_excludes_label_order(lines):
    window, label = split_last_prior(lines)
    assert not window["order_id"].isin(label["order_id"]).any()
    assert 7 not in set(window["product_id"])
    # every earlier order is kept, and nothing is lost or duplicated
    assert len(window) + len(label) == len(lines)
    last = window["user_id"].map(lines.groupby("user_id")["order_number"].max())
    assert (window["order_number"] < last).all()


def test_assert_prior_only_catches_final_orders():
    orders = pd.DataFrame({"order_id": [1, 2, 3], "eval_set": ["prior", "train", "test"]})
    assert_prior_only([1], orders)  # prior: fine
    for leaked in ([1, 2], [3]):
        with pytest.raises(AssertionError, match="final-order"):
            assert_prior_only(leaked, orders)
