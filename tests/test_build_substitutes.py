"""Tests for downcast_counts in src/build_substitutes.py."""
import numpy as np
import pandas as pd
import pytest

from src.build_substitutes import downcast_counts

INT32_MAX = np.iinfo("int32").max


def counts(**overrides):
    """A tiny pairs table with valid counts; override one column to break it."""
    df = pd.DataFrame({
        "k_m": np.array([5, 7], dtype="int64"),
        "n_m": np.array([30, 40], dtype="int64"),
        "n_b": np.array([100, 200], dtype="int64"),
        "k_b": np.array([0.0, 3.0]),  # float, as after fillna
        "lift": [2.0, 1.5],
    })
    return df.assign(**overrides)


def test_valid_counts_become_int32_unchanged():
    df = counts()
    out = downcast_counts(df)
    for c in ("k_m", "n_m", "n_b", "k_b"):
        assert out[c].dtype == "int32"
        assert (out[c] == df[c]).all()
    assert out["lift"].dtype == "float64"  # other columns untouched


def test_count_above_int32_max_raises():
    df = counts(n_b=np.array([100, INT32_MAX + 1], dtype="int64"))
    with pytest.raises(ValueError, match="int32 range"):
        downcast_counts(df)


def test_non_whole_count_raises():
    df = counts(k_b=[0.0, 2.5])
    with pytest.raises(ValueError, match="non-whole"):
        downcast_counts(df)
