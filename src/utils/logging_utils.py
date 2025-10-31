import logging
from pathlib import Path


def setup_logging(level="INFO", log_dir="logs", log_name="train.log"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_name

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.info(
        f"Логирование инициализировано. Уровень: {level}. Файл: {log_path}"
    )
