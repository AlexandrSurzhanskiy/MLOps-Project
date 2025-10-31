import torch
import json
import logging
import inspect
from pathlib import Path


def save_model(
    model, output_dir: str, metrics: dict = None, config: dict = None
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)

    if config:
        valid_args = inspect.signature(model.__class__.__init__).parameters
        filtered_config = {k: v for k, v in config.items() if k in valid_args}

        with open(output_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(filtered_config, f, indent=2)

    if metrics:
        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    logging.info(f"Модель сохранена в {output_path}")


def load_model(model_class, model_dir, data_path=None):
    import pandas as pd
    import inspect
    import torch
    import json
    import logging
    from pathlib import Path

    model_dir = Path(model_dir)
    model_path = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Не найден файл модели: {model_path}")

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        logging.warning("config.json не найден, создаётся пустой конфиг")

    if data_path:
        df = pd.read_parquet(data_path)
        if "n_users" not in config:
            config["n_users"] = int(df["user_idx"].max()) + 1
        if "n_items" not in config:
            config["n_items"] = int(df["item_idx"].max()) + 1
        logging.warning(
            "n_users/n_items восстановлены из данных при отсутствии в конфиге"
        )

    valid_args = inspect.signature(model_class.__init__).parameters
    filtered_config = {k: v for k, v in config.items() if k in valid_args}

    logging.info(f"Конфиг модели: {filtered_config}")

    model = model_class(**filtered_config)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logging.info(f"Модель успешно загружена из {model_dir}")
    return model
