import argparse
import logging
from pathlib import Path

from src.utils import setup_logging
from src.data.preprocess import preprocess_events
from src.data.split import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Предобработка events.csv и подготовка train/test parquet")
    parser.add_argument("--input", type=str, required=True, help="Путь к data/raw/events.csv")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Директория для сохранения data/processed (interactions/train/test)",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Доля теста")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    setup_logging(level="INFO")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = preprocess_events(
        raw_path=args.input,
        output_path=str(out_dir / "interactions.parquet"),
    )

    train_path, test_path = split_dataset(
        processed_path=str(interactions_path),
        output_dir=str(out_dir),
        test_size=args.test_size,
        seed=args.seed,
    )

    logging.info(f"Готово: {interactions_path}")
    logging.info(f"Train: {train_path}")
    logging.info(f"Test : {test_path}")


if __name__ == "__main__":
    main()
