"""Error analysis: inspect the worst-performing test queries for the best model."""

from pathlib import Path
import pandas as pd
from train_eval import compute_ndcg_at_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def analyze_worst_queries(
    pred_path: Path = PROCESSED_DIR / "test_predictions.parquet",
    score_col: str = "score_hgb",
    top_n: int = 8,
) -> pd.DataFrame:
    """Compute per-query NDCG@10 and print the worst-performing queries."""
    df = pd.read_parquet(pred_path)

    records = []
    for qid, grp in df.groupby("query_id", sort=False):
        rels = grp["relevance"].to_numpy()
        scores = grp[score_col].to_numpy()
        ndcg = compute_ndcg_at_k(rels, scores, k=10)

        top_pred_row = grp.sort_values(score_col, ascending=False).iloc[0]
        exact_rows = grp[grp["relevance"] == 2]
        exact_example = (
            exact_rows.iloc[0]["product_name"]
            if len(exact_rows) > 0
            else "(No Exact-labeled product in test candidates)"
        )
        exact_class = exact_rows.iloc[0]["product_class"] if len(exact_rows) > 0 else "N/A"

        records.append(
            {
                "query_id": qid,
                "query": grp["query"].iloc[0],
                "num_candidates": len(grp),
                "num_exact": int((rels == 2).sum()),
                "ndcg@10": ndcg,
                "top_pred_product": top_pred_row["product_name"],
                "top_pred_rel": int(top_pred_row["relevance"]),
                "top_pred_class": top_pred_row["product_class"],
                "actual_exact_product": exact_example,
                "actual_exact_class": exact_class,
            }
        )

    summary = pd.DataFrame(records).sort_values("ndcg@10", ascending=True).reset_index(drop=True)
    worst = summary.head(top_n)

    print(f"=== Worst {top_n} Test Queries by NDCG@10 ({score_col}) ===\n")
    for idx, row in worst.iterrows():
        print(
            f"[{idx + 1}] Query (id={row['query_id']}): '{row['query']}' | "
            f"NDCG@10={row['ndcg@10']:.4f} | "
            f"Candidates={row['num_candidates']} (Exact={row['num_exact']})"
        )
        print(
            f"    Top Predicted (rel={row['top_pred_rel']}, class={row['top_pred_class']}): "
            f"{row['top_pred_product']}"
        )
        print(
            f"    Actual Exact  (class={row['actual_exact_class']}): "
            f"{row['actual_exact_product']}\n"
        )

    return worst


if __name__ == "__main__":
    analyze_worst_queries()

