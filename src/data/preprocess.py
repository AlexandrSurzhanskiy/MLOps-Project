import logging
from pathlib import Path

import pandas as pd


def check_data_format(df: pd.DataFrame, required_cols=("user_idx", "item_idx", "label")) -> bool:
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
    raw_path: str = "data/raw/events.csv",
    output_path: str = "data/processed/interactions.parquet",
) -> Path:
    raw_path = Path(raw_path).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Не найден файл: {raw_path}")

    output_path = Path(output_path).expanduser()
    if output_path.suffix == "":
        output_path = output_path / "interactions.parquet"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Загрузка данных из {raw_path} ...")
    df = pd.read_csv(raw_path)
    logging.info(f"Исходная форма: {df.shape}")

    required_raw = {"event", "visitorid", "itemid"}
    missing = required_raw - set(df.columns)
    if missing:
        raise ValueError(f"В events.csv нет колонок: {sorted(missing)}. Есть: {df.columns.tolist()}")

    df = df[df["event"].isin(["view", "transaction"])].copy()
    df["label"] = (df["event"] == "transaction").astype(int)

    users_sorted = sorted(df["visitorid"].dropna().unique().tolist())
    items_sorted = sorted(df["itemid"].dropna().unique().tolist())
    user2idx = {u: i for i, u in enumerate(users_sorted)}
    item2idx = {it: j for j, it in enumerate(items_sorted)}

    df["user_idx"] = df["visitorid"].map(user2idx).astype("int64")
    df["item_idx"] = df["itemid"].map(item2idx).astype("int64")

    df = df.groupby(["user_idx", "item_idx"], as_index=False).agg({"label": "max"})

    check_data_format(df, required_cols=("user_idx", "item_idx", "label"))

    df.to_parquet(output_path, index=False)
    logging.info(f"Обработанный датасет сохранён в {output_path} ({len(df):,} записей)")

    return output_path
