import argparse
import pandas as pd
import logging
from pathlib import Path
from src.utils import setup_logging
from src.evaluation import compute_metrics, generate_report


def find_latest_prediction(version_name: str, base_dir="MLOps/predictions") -> Path:
    version_dir = Path(base_dir) / version_name
    if not version_dir.exists():
        raise FileNotFoundError(f"Директория {version_dir} не найдена")

    csv_files = sorted(
        version_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(
            f"В {version_dir} нет ни одного .csv файла с предсказаниями"
        )

    latest = csv_files[0]
    logging.info(f"Найден последний файл предсказаний: {latest}")
    return latest


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Либо путь к CSV, либо имя модели (например: recsys_nn_v1)",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Путь к parquet с целевой переменной",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="MLOps/reports",
        help="Каталог для сохранения отчётов",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    logging.info("Запуск оценки качества модели")

    pred_path = Path(args.predictions)
    if pred_path.suffix == "":
        pred_path = find_latest_prediction(args.predictions)

    elif pred_path.is_dir():
        pred_path = find_latest_prediction(pred_path.name, base_dir=pred_path.parent)

    logging.info(f"Используем файл c предсказаниями: {pred_path}")

    preds_df = pd.read_csv(pred_path)
    gt_df = pd.read_parquet(args.ground_truth)
    logging.info(f"Загружено {len(preds_df):,} скоров и {len(gt_df):,} таргетов")

    for col in ["target", "label", "purchased", "event", "click"]:
        if col in gt_df.columns:
            target_col = col
            break
    else:
        raise ValueError(
            f"Не удалось найти колонку таргета. Доступные: {gt_df.columns.tolist()}"
        )

    logging.info(f"Используется колонка таргета: {target_col}")

    merged = preds_df.merge(
        gt_df[["user_idx", "item_idx", target_col]],
        on=["user_idx", "item_idx"],
        how="inner",
        suffixes=("_pred", "_true"),
    )

    if f"{target_col}_true" in merged.columns:
        merged = merged.rename(columns={f"{target_col}_true": "target"})
    elif target_col in merged.columns:
        merged = merged.rename(columns={target_col: "target"})
    else:
        raise ValueError(
            f"После объединения не найдена колонка 'target'. Есть: {merged.columns.tolist()}"
        )

    logging.info(f"Совмещено {len(merged):,} строк для оценки")

    metrics = compute_metrics(merged["target"].values, merged["predicted_score"].values)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    report_config = {
        "model_name": Path(pred_path).parents[0].name,
        "predictions_file": str(pred_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    generate_report(
        metrics, config=report_config, output_dir=str(output_dir), fmt="txt"
    )

    logging.info(f"Метрики: {metrics}")
    logging.info(f"Отчёт сохранён в {output_dir}")
    logging.info("Оценка завершена успешно!")


if __name__ == "__main__":
    main()
