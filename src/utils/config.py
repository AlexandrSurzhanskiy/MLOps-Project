import yaml
from pathlib import Path


def load_config(path: str):
    path = Path(path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).resolve().parents[2]
    def make_absolute(d):
        if isinstance(d, dict):
            return {k: make_absolute(v) for k, v in d.items()}
        elif isinstance(d, str) and ("/" in d or "\\" in d):
            p = Path(d)
            return str((root / p).resolve()) if not p.is_absolute() else str(p)
        else:
            return d

    cfg = make_absolute(cfg)
    return cfg


def merge_dicts(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in result:
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result
