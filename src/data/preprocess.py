import pandas as pd
import logging
from pathlib import Path


def check_data_format(df, required_cols=("user_idx", "item_idx", "label")):
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Ожидается pandas DataFrame")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing}")

    if not pd.api.types.is_integer_dtype(df["user_idx"]):
        raise TypeError("user_idx должен быть целым")
    if not pd.api.types.is_integer_dtype(df["item_idx"]):
        raise TypeError("item_idx должен быть целым")
    if not pd.api.types.is_numeric_dtype(df["label"]):
        raise TypeError("label должен быть числовым")

    if df.isna().any().any():
        raise ValueError("Данные содержат пропущенные значения")

    return True


def preprocess_events(
    raw_path="data/raw/events.csv",
    output_path="data/processed/interactions.parquet",
):
    logging.info(f"Загрузка данных из {raw_path} ...")
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Не найден файл: {raw_path}")

    df = pd.read_csv(raw_path)
    logging.info(f"Исходная форма: {df.shape}")

    df = df[df["event"].isin(["view", "transaction"])].copy()

    df["label"] = (df["event"] == "transaction").astype(int)

    user2idx = {u: i for i, u in enumerate(df["visitorid"].unique())}
    item2idx = {i: j for j, i in enumerate(df["itemid"].unique())}
    df["user_idx"] = df["visitorid"].map(user2idx)
    df["item_idx"] = df["itemid"].map(item2idx)

    df = df.groupby(["user_idx", "item_idx"], as_index=False).agg(
        {"label": "max"}
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(
        f"Обработанный датасет сохранён в {output_path} ({len(df)} записей)"
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    preprocess_events(
        "data/raw/events.csv", "data/processed/interactions.parquet"
    )
