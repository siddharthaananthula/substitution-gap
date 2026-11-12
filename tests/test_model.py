"""Tests for src/model.py helpers and per-event scoring (synthetic data)."""
import numpy as np
import pandas as pd

from src.evaluate import score_events
from src.model import customer_split, group_sizes, model_candidates, positive_events_only


def rows():
    return pd.DataFrame({
        "event_id": [0, 0, 1, 1, 1, 2, 2],
        "user_id":  [5, 5, 6, 6, 6, 7, 7],
        "b":        [10, 11, 10, 12, 13, 10, 11],
        "label":    [0, 1, 0, 0, 0, 1, 0],
    })


def test_positive_events_only_drops_all_zero_groups():
    kept = positive_events_only(rows())
    assert sorted(kept["event_id"].unique()) == [0, 2]
    assert len(kept) == 4


def test_customer_split_is_by_customer_and_fixed():
    r = pd.DataFrame({"user_id": np.repeat(np.arange(100), 3)})
    m1, m2 = customer_split(r), customer_split(r)
    assert (m1 == m2).all()  # fixed seed
    assert r.loc[m1, "user_id"].nunique() == 20
    assert not set(r.loc[m1, "user_id"]) & set(r.loc[~m1, "user_id"])


def test_group_sizes():
    assert group_sizes(rows()).tolist() == [2, 3, 2]


def test_model_candidates_rank_by_score_then_id():
    top = model_candidates(rows(), np.array([0.1, 0.9, 0.5, 0.5, 0.2, 0.3, 0.7]), k=2)
    got = {e: g.sort_values("rank")["b"].tolist() for e, g in top.groupby("event_id")}
    assert got == {0: [11, 10], 1: [10, 12], 2: [11, 10]}  # event 1: tie -> lower id first


def test_score_events_per_event_matches_per_a():
    events = pd.DataFrame({"event_id": [0, 1], "order_id": [100, 101], "a": [1, 1],
                           "n_loose": [1, 0], "n_strict": [1, 0], "n_aisle": [1, 0]})
    truth = pd.DataFrame({"order_id": [100], "b": [3]})
    groups = {"strict": pd.Series({1: 1, 2: 1, 3: 1}), "aisle": pd.Series({1: 9, 2: 9, 3: 9})}
    by_a = pd.DataFrame({"a": [1, 1], "rank": [1, 2], "b": [2, 3]})
    by_event = pd.DataFrame({"event_id": [0, 0, 1, 1], "rank": [1, 2, 1, 2], "b": [2, 3, 2, 3]})
    pd.testing.assert_frame_equal(score_events(events, by_a, truth, groups),
                                  score_events(events, by_event, truth, groups))
