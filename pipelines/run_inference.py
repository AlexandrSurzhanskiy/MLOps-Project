import argparse
import torch
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from src.utils import setup_logging, load_model
from src.models.recsys_nn import RecSysNN


def main():
    parser = argparse.ArgumentParser(
        description="Запуск инференса обученной рекомендательной модели"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Путь к директории с сохранённой моделью",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Путь к parquet-файлу с парами user-item для расчёта предсказаний",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="MLOps/predictions",
        help="Корневая директория для сохранения предсказаний",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Запуск инференса на устройстве: {device}")

    model_dir = Path(args.model_dir)
    model_version = model_dir.name
    logging.info(f"Используется модель версии: {model_version}")

    model = load_model(RecSysNN, model_dir, data_path=args.data)
    model.to(device)
    logging.info(f"Модель успешно загружена из {model_dir}")

    df = pd.read_parquet(args.data)
    logging.info(f"Загружено {len(df):,} записей из {args.data}")

    users = torch.tensor(df["user_idx"].values, dtype=torch.long).to(device)
    items = torch.tensor(df["item_idx"].values, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(users, items).cpu().numpy()

    df["predicted_score"] = preds

    root_dir = Path(args.output)
    version_dir = root_dir / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = version_dir / f"predictions_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    logging.info(f"Предсказания сохранены: {output_path}")
    logging.info("Инференс завершён успешно!")


if __name__ == "__main__":
    main()
