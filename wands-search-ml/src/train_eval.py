"""Feature engineering, model training, and ranking evaluation on WANDS."""

from pathlib import Path
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

SQL_FEATURE_COLS = [
    "title_word_count",
    "desc_word_count",
    "category_product_count",
    "query_word_count",
]

FEATURE_COLS = [
    "tfidf_title_sim",
    "tfidf_desc_sim",
    "word_overlap_ratio",
    "exact_match",
    "category_match",
    *SQL_FEATURE_COLS,
]

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric word tokenization."""
    return set(_WORD_RE.findall(text.lower()))


def _rowwise_tfidf_cosine(
    vectorizer: TfidfVectorizer,
    left_texts: pd.Series,
    right_texts: pd.Series,
) -> np.ndarray:
    """Compute row-wise cosine similarity between two text Series using a fitted TfidfVectorizer."""
    left_mat = vectorizer.transform(left_texts.fillna(""))
    right_mat = vectorizer.transform(right_texts.fillna(""))
    # TfidfVectorizer outputs L2-normalized rows by default, so dot product == cosine similarity.
    sims = np.asarray(left_mat.multiply(right_mat).sum(axis=1)).ravel()
    return sims.astype(np.float64)


def build_feature_matrix(
    df: pd.DataFrame,
    title_vec: TfidfVectorizer,
    desc_vec: TfidfVectorizer,
) -> pd.DataFrame:
    """Compute lexical, TF-IDF, and SQL features for each (query, product) pair."""
    feats = pd.DataFrame(index=df.index)
    feats["tfidf_title_sim"] = _rowwise_tfidf_cosine(title_vec, df["query"], df["product_name"])
    feats["tfidf_desc_sim"] = _rowwise_tfidf_cosine(desc_vec, df["query"], df["product_description"])

    overlap_ratios = []
    exact_matches = []
    category_matches = []

    for q, title, pclass in zip(df["query"], df["product_name"], df["product_class"]):
        q_lower = q.lower().strip()
        t_lower = title.lower()
        q_words = _tokenize(q_lower)
        t_words = _tokenize(t_lower)
        c_words = _tokenize(pclass.lower())

        overlap_ratios.append(len(q_words & t_words) / max(len(q_words), 1))
        exact_matches.append(1 if (q_lower and q_lower in t_lower) else 0)
        category_matches.append(1 if (q_words & c_words) else 0)

    feats["word_overlap_ratio"] = np.asarray(overlap_ratios, dtype=np.float64)
    feats["exact_match"] = np.asarray(exact_matches, dtype=np.int64)
    feats["category_match"] = np.asarray(category_matches, dtype=np.int64)

    for col in SQL_FEATURE_COLS:
        feats[col] = df[col].astype(np.float64)

    return feats[FEATURE_COLS]


def fit_vectorizers(train_df: pd.DataFrame) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fit TF-IDF vectorizers on the training split only to prevent data leakage."""
    title_corpus = pd.concat([train_df["query"], train_df["product_name"]]).dropna().unique()
    desc_corpus = pd.concat([train_df["query"], train_df["product_description"]]).dropna().unique()

    title_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    title_vec.fit(title_corpus)

    desc_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    desc_vec.fit(desc_corpus)

    return title_vec, desc_vec


def compute_ndcg_at_k(relevances: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k for a single query's candidate products."""
    if len(relevances) == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")[:k]
    ranked_rel = relevances[order]
    discounts = np.log2(np.arange(len(ranked_rel)) + 2.0)
    dcg = float(np.sum(((2.0 ** ranked_rel) - 1.0) / discounts))

    ideal_order = np.argsort(-relevances, kind="mergesort")[:k]
    ideal_rel = relevances[ideal_order]
    ideal_discounts = np.log2(np.arange(len(ideal_rel)) + 2.0)
    idcg = float(np.sum(((2.0 ** ideal_rel) - 1.0) / ideal_discounts))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_mrr_at_k(relevances: np.ndarray, scores: np.ndarray, k: int = 10, target_rel: int = 2) -> float:
    """Compute MRR@k where a hit is a product with relevance >= target_rel (default: Exact=2)."""
    if len(relevances) == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")[:k]
    ranked_rel = relevances[order]
    hits = np.where(ranked_rel >= target_rel)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def evaluate_ranking(df: pd.DataFrame, score_col: str, k: int = 10) -> dict[str, float]:
    """Compute mean NDCG@k and MRR@k across all queries in df."""
    ndcgs = []
    mrrs = []
    for _, grp in df.groupby("query_id", sort=False):
        rels = grp["relevance"].to_numpy()
        scores = grp[score_col].to_numpy()
        ndcgs.append(compute_ndcg_at_k(rels, scores, k=k))
        mrrs.append(compute_mrr_at_k(rels, scores, k=k, target_rel=2))
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"mrr@{k}": float(np.mean(mrrs)),
    }


def expected_relevance_from_proba(probs: np.ndarray) -> np.ndarray:
    """Convert class probabilities [P(0), P(1), P(2)] into expected relevance in [0, 2]."""
    classes = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    return probs @ classes


def plot_ndcg_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    """Save a bar chart comparing NDCG@10 across the 3 models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#7f8c8d", "#3498db", "#2ecc71"]
    bars = ax.bar(results_df["Model"], results_df["NDCG@10"], color=colors, width=0.55)

    ax.set_ylim(0, min(1.0, results_df["NDCG@10"].max() * 1.18))
    ax.set_ylabel("NDCG@10")
    ax.set_title("WANDS Test Set Ranking Performance (NDCG@10)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("Fitting TF-IDF vectorizers on train split only...")
    title_vec, desc_vec = fit_vectorizers(train_df)

    print("Building feature matrices...")
    X_train = build_feature_matrix(train_df, title_vec, desc_vec)
    X_val = build_feature_matrix(val_df, title_vec, desc_vec)
    X_test = build_feature_matrix(test_df, title_vec, desc_vec)

    y_train = train_df["relevance"].to_numpy()
    y_test = test_df["relevance"].to_numpy()

    assert not X_train.isna().any().any(), "NaNs found in X_train!"
    assert not X_val.isna().any().any(), "NaNs found in X_val!"
    assert not X_test.isna().any().any(), "NaNs found in X_test!"

    # Model 1: Unsupervised TF-IDF Title Cosine Similarity
    test_eval = test_df[["query_id", "query", "query_word_count", "product_id", "product_name", "product_class", "relevance"]].copy()
    test_eval["score_tfidf"] = X_test["tfidf_title_sim"].to_numpy()

    # Model 2: LogisticRegression
    print("Training LogisticRegression...")
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    lr_model.fit(X_train, y_train)
    lr_test_probs = lr_model.predict_proba(X_test)
    lr_test_pred = lr_model.predict(X_test)
    test_eval["score_logreg"] = expected_relevance_from_proba(lr_test_probs)

    # Model 3: HistGradientBoostingClassifier
    print("Training HistGradientBoostingClassifier...")
    hgb_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
    )
    hgb_model.fit(X_train, y_train)
    hgb_test_probs = hgb_model.predict_proba(X_test)
    hgb_test_pred = hgb_model.predict(X_test)
    test_eval["score_hgb"] = expected_relevance_from_proba(hgb_test_probs)

    # Evaluate all 3 models on test set
    models = [
        ("TF-IDF Title (Unsupervised)", "score_tfidf", None, None),
        ("LogisticRegression", "score_logreg", lr_test_pred, test_eval["score_logreg"].to_numpy()),
        ("HistGradientBoosting", "score_hgb", hgb_test_pred, test_eval["score_hgb"].to_numpy()),
    ]

    rows = []
    for name, score_col, y_pred_cls, y_pred_cont in models:
        metrics = evaluate_ranking(test_eval, score_col, k=10)
        rmse = root_mean_squared_error(y_test, y_pred_cont) if y_pred_cont is not None else np.nan
        macro_f1 = f1_score(y_test, y_pred_cls, average="macro") if y_pred_cls is not None else np.nan
        rows.append(
            {
                "Model": name,
                "NDCG@10": metrics["ndcg@10"],
                "MRR@10": metrics["mrr@10"],
                "RMSE": rmse,
                "Macro-F1": macro_f1,
            }
        )

    results_df = pd.DataFrame(rows)
    print("\n=== Test Set Model Comparison ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Query length breakdown: Short (1-2 words) vs Long (3+ words)
    short_df = test_eval[test_eval["query_word_count"] <= 2]
    long_df = test_eval[test_eval["query_word_count"] >= 3]
    print(
        f"\n=== Query Length Breakdown (Short: {short_df['query_id'].nunique()} queries, "
        f"Long: {long_df['query_id'].nunique()} queries) ==="
    )
    len_rows = []
    for name, score_col, _, _ in models:
        s_metrics = evaluate_ranking(short_df, score_col, k=10)
        l_metrics = evaluate_ranking(long_df, score_col, k=10)
        len_rows.append(
            {
                "Model": name,
                "Short NDCG@10 (1-2w)": s_metrics["ndcg@10"],
                "Long NDCG@10 (3+w)": l_metrics["ndcg@10"],
                "Short MRR@10": s_metrics["mrr@10"],
                "Long MRR@10": l_metrics["mrr@10"],
            }
        )
    print(pd.DataFrame(len_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save artifacts for plotting and error analysis
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(ARTIFACTS_DIR / "metrics_summary.csv", index=False)
    plot_ndcg_comparison(results_df, ARTIFACTS_DIR / "ndcg_comparison.png")
    test_eval.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)
    print(f"\nSaved plot to {ARTIFACTS_DIR / 'ndcg_comparison.png'}")
    print(f"Saved test predictions to {PROCESSED_DIR / 'test_predictions.parquet'}")

    # Save trained models, vectorizers, catalog, and ground-truth labels for app.py
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(title_vec, MODELS_DIR / "title_vec.joblib")
    joblib.dump(desc_vec, MODELS_DIR / "desc_vec.joblib")
    joblib.dump(hgb_model, MODELS_DIR / "hgb_model.joblib")
    joblib.dump(lr_model, MODELS_DIR / "logreg_model.joblib")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    catalog_cols = [
        "product_id",
        "product_name",
        "product_description",
        "product_class",
        "title_word_count",
        "desc_word_count",
        "category_product_count",
    ]
    catalog_df = all_df[catalog_cols].drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    catalog_df.to_parquet(MODELS_DIR / "catalog.parquet", index=False)

    gt_df = all_df[["query", "product_id", "label_text", "relevance"]].copy()
    gt_df["query_norm"] = gt_df["query"].str.lower().str.strip()
    gt_df = gt_df.drop(columns=["query"]).drop_duplicates(subset=["query_norm", "product_id"])
    gt_df.to_parquet(MODELS_DIR / "ground_truth.parquet", index=False)
    print(f"Saved models and catalog ({len(catalog_df):,} products) to {MODELS_DIR}")


if __name__ == "__main__":
    main()

