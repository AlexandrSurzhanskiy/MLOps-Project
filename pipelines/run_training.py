import argparse
import torch
import logging
import pandas as pd
from pathlib import Path
import json
import hashlib
import subprocess
from datetime import datetime
import mlflow
import mlflow.pytorch

from src.utils import load_config, setup_logging
from src.utils.io import save_model
from src.data.dataset import get_dataloaders
from src.models import get_model
from src.training import train_model, validate
from src.evaluation import generate_report


def _md5_file(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "NA"
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def _safe_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "NA"


def _extract_dvc_out_hashes(lock_path: str = "dvc.lock") -> dict:
    p = Path(lock_path)
    if not p.exists():
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    lock = yaml.safe_load(p.read_text(encoding="utf-8"))
    stages = lock.get("stages", {})
    out_hashes = {}

    for stage_name, stage in stages.items():
        outs = stage.get("outs", [])
        for o in outs:
            path = o.get("path")
            md5 = o.get("md5") or o.get("etag")
            if path and md5:
                out_hashes[f"{stage_name}:{path}"] = md5

    return out_hashes


def main():
    parser = argparse.ArgumentParser(description="Train recommender system model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"))

    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "recsys-retailrocket")
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    run_prefix = cfg.get("mlflow", {}).get("run_name_prefix", "train")
    run_name = f"{run_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Используется устройство: {device}")

    train_path = cfg["data"]["train_path"]
    test_path = cfg["data"]["test_path"]

    if not train_path or not test_path:
        raise ValueError("В конфиге не заданы пути train_path/test_path")

    logging.info(f"Загрузка датасета: {train_path}")

    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run(run_name=run_name):
        flat_cfg = _flatten_dict(cfg)
        params = {k: v for k, v in flat_cfg.items() if isinstance(v, (str, int, float, bool)) or v is None}
        mlflow.log_params(params)

        mlflow.set_tag("git_commit", _safe_git_commit())
        mlflow.set_tag("device", device)

        if Path("dvc.lock").exists():
            mlflow.log_artifact("dvc.lock", artifact_path="dvc")
            mlflow.set_tag("dvc_lock_md5", _md5_file("dvc.lock"))

            dvc_out_hashes = _extract_dvc_out_hashes("dvc.lock")
            if dvc_out_hashes:
                tmp = Path("dvc_out_hashes.json")
                tmp.write_text(
                    json.dumps(dvc_out_hashes, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(tmp), artifact_path="dvc")
                mlflow.set_tag(
                    "dvc_data_hash",
                    hashlib.md5(json.dumps(dvc_out_hashes, sort_keys=True).encode("utf-8")).hexdigest(),
                )
                tmp.unlink(missing_ok=True)

        if Path("dvc.yaml").exists():
            mlflow.log_artifact("dvc.yaml", artifact_path="dvc")

        mlflow.log_artifact(args.config, artifact_path="configs")

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

        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

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

        model_dir = Path(cfg["training"]["output_dir"])
        if model_dir.exists():
            mlflow.log_artifacts(str(model_dir), artifact_path="saved_model")

        mlflow.pytorch.log_model(trained_model, artifact_path="mlflow_model")

        weights_path = model_dir / "pytorch_model.bin"
        if weights_path.exists():
            mlflow.set_tag("model_weights_md5", _md5_file(str(weights_path)))

        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        generate_report(metrics, cfg, output_dir=str(reports_dir), fmt="txt")

        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")

        logging.info("Обучение завершено успешно!")


if __name__ == "__main__":
    main()
