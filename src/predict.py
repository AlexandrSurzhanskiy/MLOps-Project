import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.recsys_nn import RecSysNN
from src.utils.io import load_model


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError("Поддерживаются только форматы .parquet или .csv")


def _save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return

    df.to_csv(path, index=False)


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"user_idx", "item_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В данных нет обязательных колонок {sorted(missing)}. Есть: {df.columns.tolist()}")


def predict_scores(model: torch.nn.Module, df: pd.DataFrame, device: str, batch_size: int) -> np.ndarray:
    users_all = df["user_idx"].to_numpy()
    items_all = df["item_idx"].to_numpy()

    preds_parts = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))

            users = torch.tensor(users_all[start:end], dtype=torch.long, device=device)
            items = torch.tensor(items_all[start:end], dtype=torch.long, device=device)

            p = model(users, items)
            p = p.detach().cpu().view(-1).numpy()
            preds_parts.append(p)

    return np.concatenate(preds_parts, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Инференс рекомендательной модели: читает (user_idx, item_idx) и сохраняет predicted_score"
    )
    parser.add_argument("--input_path", type=str, required=True, help="Путь к входному файлу (.parquet или .csv)")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Путь для сохранения предсказаний (.csv или .parquet)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(Path("models") / "recsys_nn_v1"),
        help="Путь к директории модели (по умолчанию: models/recsys_nn_v1)",
    )
    parser.add_argument("--batch_size", type=int, default=8192, help="Размер батча при инференсе")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Устройство: {device}")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_dir = Path(args.model_dir)

    df = _read_input(input_path)
    logging.info(f"Загружено строк: {len(df):,} из {input_path}")

    _validate_columns(df)

    model = load_model(RecSysNN, model_dir, data_path=None)
    model.to(device)
    logging.info(f"Модель загружена: {model_dir}")

    preds = predict_scores(model, df, device=device, batch_size=args.batch_size)

    out_df = df.copy()
    out_df["predicted_score"] = preds

    _save_output(out_df, output_path)
    logging.info(f"Готово! Сохранено: {output_path}")


if __name__ == "__main__":
    main()
