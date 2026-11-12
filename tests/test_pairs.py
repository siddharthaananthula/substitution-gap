"""Tests for src/pairs.py on small synthetic data (never the real dataset)."""
import numpy as np
import pandas as pd
import pytest

from src.pairs import (
    MIN_ORDERS,
    MIN_SHARE,
    add_lift_lower,
    compute_lift,
    find_regulars,
    orders_per_user,
    run_batches,
)

FILLER = 99  # in every order, so no order is ever empty


def make_lines(baskets):
    """Build prior lines from {user_id: [basket, ...]} with real dtypes.

    `reordered` is 1 if the customer bought the product in an earlier order.
    """
    rows, order_id = [], 0
    for user, orders in baskets.items():
        seen = set()
        for number, basket in enumerate(orders, start=1):
            order_id += 1
            for product in basket:
                rows.append((order_id, product, int(product in seen), user, number))
            seen.update(basket)
    lines = pd.DataFrame(
        rows, columns=["order_id", "product_id", "reordered", "user_id", "order_number"]
    )
    return lines.astype({"order_id": "int32", "product_id": "int32", "reordered": "int8",
                         "user_id": "int32", "order_number": "int16"})


def totals_for(lines, n_batches):
    reg = find_regulars(lines, orders_per_user(lines))
    return run_batches(lines, reg, n_batches=n_batches)


# --- batching ---------------------------------------------------------------

@pytest.fixture
def random_lines():
    """15 customers, 12-16 orders each, a few habitual products plus random new ones."""
    rng = np.random.default_rng(0)
    baskets = {}
    for user in range(1, 16):
        orders = []
        for _ in range(rng.integers(12, 17)):
            habits = [p for p in (1, 2, 3, 4) if rng.random() < 0.7]
            extras = rng.choice(np.arange(10, 60), size=rng.integers(0, 4), replace=False)
            orders.append([FILLER, *habits, *extras.tolist()])
        baskets[user] = orders
    return make_lines(baskets)


def test_batched_totals_equal_single_batch(random_lines):
    one = totals_for(random_lines, n_batches=1)
    three = totals_for(random_lines, n_batches=3)
    for key in ("n_m", "k_m", "n_b", "k_b"):
        assert len(one[key]) > 0, f"{key} is empty, test would be vacuous"
        pd.testing.assert_series_equal(
            one[key].sort_index(), three[key].sort_index(), check_names=False
        )


# --- planted substitute ------------------------------------------------------

A, SUB, UNRELATED = 1, 2, 3
N_ORDERS = 14
MISSING = (3, 4)  # order numbers where A is missing
# Orders that are events for A: missing ones, plus bought-again ones (order 1 is A's first buy)
EVENT_ORDERS = [n for n in range(2, N_ORDERS + 1)]


@pytest.fixture
def planted_pairs():
    """A is missing in orders 3-4 for every customer.

    SUB is bought for the first time only in order 3 (A missing).
    UNRELATED is bought for the first time in an order picked in rotation over
    all event orders, so it lands in missing and bought orders at the same rate.
    """
    baskets = {}
    for user in range(1, 40):  # 39 customers = 3 full rotations over 13 event orders
        unrelated_at = EVENT_ORDERS[(user - 1) % len(EVENT_ORDERS)]
        orders = []
        for n in range(1, N_ORDERS + 1):
            basket = [FILLER]
            if n not in MISSING:
                basket.append(A)
            if n == 3:
                basket.append(SUB)
            if n == unrelated_at:
                basket.append(UNRELATED)
            orders.append(basket)
        baskets[user] = orders
    lines = make_lines(baskets)
    pairs = add_lift_lower(compute_lift(totals_for(lines, n_batches=1)))
    return pairs[pairs["a"] == A].set_index("b")


def test_planted_substitute_has_high_lift(planted_pairs):
    assert planted_pairs.loc[SUB, "lift"] > 1
    assert planted_pairs.loc[SUB, "lift_low"] > 1


def test_unrelated_product_has_lift_near_one(planted_pairs):
    assert planted_pairs.loc[UNRELATED, "lift"] == pytest.approx(1, abs=0.1)


def test_no_inf_or_nan_when_k_b_is_zero(planted_pairs):
    # SUB is never bought when A is bought, so it has no k_b row before fillna
    assert planted_pairs.loc[SUB, "k_b"] == 0
    for col in ("lift", "lift_low"):
        assert np.isfinite(planted_pairs[col]).all()


# --- regulars ----------------------------------------------------------------

def test_find_regulars_respects_thresholds():
    keep = int(np.ceil(MIN_SHARE * MIN_ORDERS))  # orders needed to reach MIN_SHARE
    baskets = {
        # exactly MIN_ORDERS orders: product 10 hits MIN_SHARE, product 11 is one short
        1: [[FILLER] + ([10] if i < keep else []) + ([11] if i < keep - 1 else [])
            for i in range(MIN_ORDERS)],
        # one order too few: nothing is a regular, even a product in every order
        2: [[FILLER, 10] for _ in range(MIN_ORDERS - 1)],
    }
    lines = make_lines(baskets)
    reg = find_regulars(lines, orders_per_user(lines))
    got = set(map(tuple, reg[["user_id", "product_id"]].to_numpy()))
    assert got == {(1, 10), (1, FILLER)}
