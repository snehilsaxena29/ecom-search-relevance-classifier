"""Load WANDS dataset, compute DuckDB SQL features, and create leakage-free splits."""

from pathlib import Path
import duckdb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "WANDS" / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQL_PATH = PROJECT_ROOT / "sql" / "features.sql"


def load_and_featurize(
    raw_dir: Path = RAW_DATA_DIR,
    sql_path: Path = SQL_PATH,
) -> pd.DataFrame:
    """Load WANDS TSV files into DuckDB and execute sql/features.sql."""
    required_files = ["query.csv", "product.csv", "label.csv"]
    for fname in required_files:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing WANDS file: {fpath}. "
                "Clone https://github.com/wayfair/WANDS.git into data/WANDS first."
            )

    sql_query = sql_path.read_text(encoding="utf-8")

    con = duckdb.connect(database=":memory:")
    try:
        # WANDS dataset files are tab-separated despite the .csv extension.
        con.execute(
            f"CREATE VIEW queries AS SELECT * FROM read_csv('{raw_dir / 'query.csv'}', delim='\\t', header=true, quote='\"');"
        )
        con.execute(
            f"CREATE VIEW products AS SELECT * FROM read_csv('{raw_dir / 'product.csv'}', delim='\\t', header=true, quote='\"');"
        )
        con.execute(
            f"CREATE VIEW labels AS SELECT * FROM read_csv('{raw_dir / 'label.csv'}', delim='\\t', header=true, quote='\"');"
        )
        df = con.execute(sql_query).df()
    finally:
        con.close()

    # Ensure numeric types and zero NaNs in text/numeric columns.
    text_cols = ["query", "query_class", "product_name", "product_description", "product_class", "label_text"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    int_cols = [
        "id",
        "query_id",
        "product_id",
        "relevance",
        "query_word_count",
        "title_word_count",
        "desc_word_count",
        "category_product_count",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df


def split_by_query(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train (70%), val (15%), test (15%) grouped by query_id."""
    splitter_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(splitter_train.split(df, groups=df["query_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    splitter_val_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_idx, test_idx = next(splitter_val_test.split(temp_df, groups=temp_df["query_id"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Sanity check: zero query_id overlap across splits.
    train_q = set(train_df["query_id"])
    val_q = set(val_df["query_id"])
    test_q = set(test_df["query_id"])
    assert train_q.isdisjoint(val_q), "Leakage between train and val!"
    assert train_q.isdisjoint(test_q), "Leakage between train and test!"
    assert val_q.isdisjoint(test_q), "Leakage between val and test!"

    return train_df, val_df, test_df


def main() -> None:
    print("Loading WANDS dataset and computing DuckDB SQL features...")
    df = load_and_featurize()
    print(f"Total labeled pairs: {len(df):,} | Unique queries: {df['query_id'].nunique():,}")
    print("Relevance distribution:\n", df["relevance"].value_counts().sort_index().to_string())

    train_df, val_df, test_df = split_by_query(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(
            f"{name:>5}: {len(split_df):>7,} rows | "
            f"{split_df['query_id'].nunique():>3} queries "
            f"({len(split_df) / len(df) * 100:.1f}%)"
        )
    print(f"Saved processed splits to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

