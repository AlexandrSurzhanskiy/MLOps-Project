import argparse
import torch
import logging
import pandas as pd
from pathlib import Path
from src.utils import load_config, setup_logging
from src.utils.io import save_model
from src.data.dataset import get_dataloaders
from src.models import get_model
from src.training import train_model, validate
from src.evaluation import generate_report


def main():
    parser = argparse.ArgumentParser(description="Train recommender system model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Используется устройство: {device}")

    train_path = cfg.get("train_path")
    test_path = cfg.get("test_path")

    logging.info(f"Загрузка датасета: {train_path}")

    train_loader, val_loader = get_dataloaders(train_path, test_path, batch_size=cfg["training"]["batch_size"])

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(test_path)

    n_users = max(train_df["user_idx"].max(), val_df["user_idx"].max()) + 1
    n_items = max(train_df["item_idx"].max(), val_df["item_idx"].max()) + 1

    model = get_model(
        model_name=cfg["model"]["name"],
        n_users=n_users,
        n_items=n_items,
        **cfg["model"]["params"],
    )

    trained_model = train_model(model, train_loader, val_loader, cfg, device=device)

    metrics = validate(trained_model, val_loader, device=device)

    config = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        **cfg["model"]["params"],
    }

    save_model(
        trained_model,
        cfg["training"]["output_dir"],
        metrics=metrics,
        config=config,
    )

    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    generate_report(metrics, cfg, output_dir=str(reports_dir), fmt="txt")

    logging.info("Обучение завершено успешно!")


if __name__ == "__main__":
    main()
