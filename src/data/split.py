import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import check_data_format


def split_dataset(
    processed_path: str = "data/processed/interactions.parquet",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    seed: int = 42,
):
    processed_path = Path(processed_path).expanduser().resolve()
    if not processed_path.exists():
        raise FileNotFoundError(f"Файл не найден: {processed_path}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_path)
    logging.info(f"Загружен {len(df):,} записей для сплита")

    check_data_format(df, required_cols=("user_idx", "item_idx", "label"))

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logging.info(f"Train: {len(train_df):,}, Test: {len(test_df):,} — сохранены в {out_dir}")

    return train_path, test_path
