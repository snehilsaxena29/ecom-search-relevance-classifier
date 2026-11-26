"""Fast pytest sanity checks for leakage, metric bounds, and feature cleanliness."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_eval import build_feature_matrix, compute_ndcg_at_k, fit_vectorizers  # noqa: E402


def test_no_query_id_leakage() -> None:
    """Ensure train, val, and test splits have strictly disjoint query_id sets."""
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=["query_id"])
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet", columns=["query_id"])
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet", columns=["query_id"])

    train_q = set(train_df["query_id"].unique())
    val_q = set(val_df["query_id"].unique())
    test_q = set(test_df["query_id"].unique())

    assert len(train_q) > 0 and len(val_q) > 0 and len(test_q) > 0
    assert train_q.isdisjoint(val_q), "query_id overlap between train and val"
    assert train_q.isdisjoint(test_q), "query_id overlap between train and test"
    assert val_q.isdisjoint(test_q), "query_id overlap between val and test"


def test_ndcg_valid_range() -> None:
    """Ensure NDCG@10 stays within [0.0, 1.0] across normal and edge cases."""
    rels = np.array([2, 1, 0, 2, 0, 1])
    perfect_scores = np.array([2.0, 1.0, 0.0, 2.0, 0.0, 1.0])
    reversed_scores = -perfect_scores
    all_zero_rels = np.zeros(6)

    ndcg_perfect = compute_ndcg_at_k(rels, perfect_scores, k=10)
    ndcg_reversed = compute_ndcg_at_k(rels, reversed_scores, k=10)
    ndcg_zeros = compute_ndcg_at_k(all_zero_rels, perfect_scores, k=10)

    assert 0.0 <= ndcg_perfect <= 1.0
    assert np.isclose(ndcg_perfect, 1.0)
    assert 0.0 <= ndcg_reversed <= 1.0
    assert ndcg_zeros == 0.0


def test_no_nans_in_features() -> None:
    """Ensure feature matrix contains no NaN or infinite values on a sample of splits."""
    train_sample = pd.read_parquet(PROCESSED_DIR / "train.parquet").head(1000)
    test_sample = pd.read_parquet(PROCESSED_DIR / "test.parquet").head(1000)

    title_vec, desc_vec = fit_vectorizers(train_sample)
    X_train = build_feature_matrix(train_sample, title_vec, desc_vec)
    X_test = build_feature_matrix(test_sample, title_vec, desc_vec)

    assert not X_train.isna().any().any(), "NaNs found in X_train"
    assert not X_test.isna().any().any(), "NaNs found in X_test"
    assert np.isfinite(X_train.to_numpy()).all(), "Non-finite values in X_train"
    assert np.isfinite(X_test.to_numpy()).all(), "Non-finite values in X_test"

