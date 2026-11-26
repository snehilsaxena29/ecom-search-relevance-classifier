"""Interactive Streamlit demo comparing TF-IDF Baseline vs. ML Ranker on WANDS."""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_eval import (  # noqa: E402
    FEATURE_COLS,
    build_feature_matrix,
    expected_relevance_from_proba,
)

PRESET_QUERIES = [
    ("nautical platters", "Short head query"),
    ("dining room tables", "ML +0.76 NDCG@10"),
    ("cake cutlery", "ML 1.00 NDCG@10"),
    ("wood coffee table set by storage", "Multi-word descriptive"),
]

BADGE_COLORS = {
    "Exact": ("#1b5e20", "#e8f5e9"),
    "Partial": ("#e65100", "#fff3e0"),
    "Irrelevant": ("#424242", "#eeeeee"),
}


@st.cache_resource
def load_models_and_index():
    """Load trained vectorizers, ML ranker, and precomputed catalog TF-IDF matrix."""
    title_vec = joblib.load(MODELS_DIR / "title_vec.joblib")
    desc_vec = joblib.load(MODELS_DIR / "desc_vec.joblib")
    hgb_model = joblib.load(MODELS_DIR / "hgb_model.joblib")
    catalog_df = pd.read_parquet(MODELS_DIR / "catalog.parquet")
    title_matrix = title_vec.transform(catalog_df["product_name"].fillna(""))
    return title_vec, desc_vec, hgb_model, catalog_df, title_matrix


@st.cache_data
def load_ground_truth() -> pd.DataFrame:
    """Load WANDS ground-truth labels keyed by normalized query and product_id."""
    return pd.read_parquet(MODELS_DIR / "ground_truth.parquet")


@st.cache_data
def load_test_metrics() -> dict[str, dict[str, float]]:
    """Load saved test-set metrics with fallback to observed values."""
    defaults = {
        "tfidf": {"ndcg@10": 0.6463, "mrr@10": 0.3950},
        "hgb": {"ndcg@10": 0.6957, "mrr@10": 0.5080},
    }
    csv_path = ARTIFACTS_DIR / "metrics_summary.csv"
    if not csv_path.exists():
        return defaults
    df = pd.read_csv(csv_path)
    out = dict(defaults)
    for _, row in df.iterrows():
        name = str(row["Model"])
        if "TF-IDF" in name:
            out["tfidf"] = {"ndcg@10": float(row["NDCG@10"]), "mrr@10": float(row["MRR@10"])}
        elif "HistGradientBoosting" in name:
            out["hgb"] = {"ndcg@10": float(row["NDCG@10"]), "mrr@10": float(row["MRR@10"])}
    return out


def search_and_rank(
    query: str,
    top_k: int = 5,
    candidate_pool: int = 100,
    resources=None,
    gt_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-filter top candidates by TF-IDF title similarity, then score with ML ranker."""
    if resources is None:
        resources = load_models_and_index()
    if gt_df is None:
        gt_df = load_ground_truth()

    title_vec, desc_vec, hgb_model, catalog_df, title_matrix = resources

    q_clean = query.strip()
    if not q_clean:
        empty = pd.DataFrame()
        return empty, empty

    # Fast sparse dot product across all catalog titles (~5-15 ms on CPU).
    q_vec = title_vec.transform([q_clean])
    sims = np.asarray((title_matrix @ q_vec.T).toarray()).ravel()

    pool_size = min(candidate_pool, len(catalog_df))
    top_idx = np.argpartition(-sims, pool_size - 1)[:pool_size]
    candidates = catalog_df.iloc[top_idx].copy().reset_index(drop=True)
    candidates["query"] = q_clean
    candidates["query_word_count"] = len(q_clean.split())

    # Compute exact 9 features on the candidate pool only.
    X_cand = build_feature_matrix(candidates, title_vec, desc_vec)
    probs = hgb_model.predict_proba(X_cand)
    ml_scores = expected_relevance_from_proba(probs)

    candidates["tfidf_score"] = X_cand["tfidf_title_sim"].to_numpy()
    candidates["ml_score"] = ml_scores
    for col in FEATURE_COLS:
        candidates[f"feat_{col}"] = X_cand[col].to_numpy()

    # Attach ground-truth labels if this query exists in WANDS.
    q_norm = q_clean.lower()
    q_gt = gt_df[gt_df["query_norm"] == q_norm][["product_id", "label_text"]]
    if len(q_gt) > 0:
        candidates = candidates.merge(q_gt, on="product_id", how="left")
    else:
        candidates["label_text"] = np.nan

    tfidf_top = candidates.sort_values("tfidf_score", ascending=False).head(top_k).reset_index(drop=True)
    ml_top = candidates.sort_values("ml_score", ascending=False).head(top_k).reset_index(drop=True)
    return tfidf_top, ml_top


def render_result_card(rank: int, row: pd.Series, score_col: str, score_label: str) -> None:
    """Render a compact product result card with optional WANDS ground-truth badge."""
    badge_html = ""
    label = row.get("label_text")
    if isinstance(label, str) and label in BADGE_COLORS:
        fg, bg = BADGE_COLORS[label]
        badge_html = (
            f"<span style='background:{bg};color:{fg};padding:2px 8px;"
            f"border-radius:12px;font-size:0.78rem;font-weight:600;margin-left:8px;'>"
            f"{label}</span>"
        )

    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0;border-radius:8px;padding:10px 12px;margin-bottom:8px;">
            <div style="font-weight:600;font-size:0.95rem;">
                #{rank} &nbsp;{row['product_name']}{badge_html}
            </div>
            <div style="color:#666;font-size:0.82rem;margin-top:4px;">
                Category: <code>{row['product_class']}</code> &nbsp;|&nbsp;
                {score_label}: <b>{row[score_col]:.4f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Wayfair WANDS — Search Relevance & Ranking Demo",
        layout="wide",
    )

    st.title("Wayfair WANDS — Search Relevance & Ranking Demo")
    st.caption(
        "Compare an unsupervised **TF-IDF Cosine Similarity** baseline against a supervised "
        "**HistGradientBoosting** ranker trained on lexical overlap, TF-IDF, and DuckDB catalog features."
    )

    metrics = load_test_metrics()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TF-IDF NDCG@10", f"{metrics['tfidf']['ndcg@10']:.4f}")
    m2.metric(
        "ML Ranker NDCG@10",
        f"{metrics['hgb']['ndcg@10']:.4f}",
        delta=f"+{metrics['hgb']['ndcg@10'] - metrics['tfidf']['ndcg@10']:.4f}",
    )
    m3.metric("TF-IDF MRR@10", f"{metrics['tfidf']['mrr@10']:.4f}")
    m4.metric(
        "ML Ranker MRR@10",
        f"{metrics['hgb']['mrr@10']:.4f}",
        delta=f"+{metrics['hgb']['mrr@10'] - metrics['tfidf']['mrr@10']:.4f}",
    )

    st.divider()

    if "query_input" not in st.session_state:
        st.session_state["query_input"] = PRESET_QUERIES[0][0]

    st.markdown("**Try a test-set preset query:**")
    preset_cols = st.columns(len(PRESET_QUERIES))
    for col, (q_text, q_desc) in zip(preset_cols, PRESET_QUERIES):
        if col.button(f"{q_text}\n({q_desc})", use_container_width=True):
            st.session_state["query_input"] = q_text

    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        query = st.text_input("Search query", key="query_input", placeholder="e.g., mid century velvet sofa")
    with ctrl_col2:
        top_k = st.slider("Top-K results", min_value=3, max_value=10, value=5)

    if not query.strip():
        st.info("Enter a search query or click one of the preset buttons above.")
        return

    tfidf_top, ml_top = search_and_rank(query, top_k=top_k)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("TF-IDF Cosine Similarity (Baseline)")
        for idx, row in tfidf_top.iterrows():
            render_result_card(idx + 1, row, "tfidf_score", "TF-IDF Sim")

    with right_col:
        st.subheader("ML Ranker (HistGradientBoosting)")
        for idx, row in ml_top.iterrows():
            render_result_card(idx + 1, row, "ml_score", "Expected Rel (0–2)")

    with st.expander("Why did the ML Ranker score these higher?"):
        feat_cols = [f"feat_{c}" for c in FEATURE_COLS]
        explain_df = ml_top[["product_name", "ml_score", *feat_cols]].copy()
        explain_df.columns = ["product_name", "ml_score", *FEATURE_COLS]
        explain_df.index = [f"#{i + 1}" for i in range(len(explain_df))]
        st.dataframe(explain_df, use_container_width=True)


if __name__ == "__main__":
    main()

