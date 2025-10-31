import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path


def split_dataset(
    processed_path="data/processed/interactions.parquet", test_size=0.2, seed=42
):
    processed_path = Path(processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(f"Файл не найден: {processed_path}")

    df = pd.read_parquet(processed_path)
    logging.info(f"Загружен {len(df):,} записей для сплита")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train_df.to_parquet("data/processed/train.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)

    logging.info(
        f"Train: {len(train_df):,}, Test: {len(test_df):,} — сохранены в data/processed/"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    split_dataset(
        processed_path="data/processed/interactions.parquet", test_size=0.2, seed=42
    )
