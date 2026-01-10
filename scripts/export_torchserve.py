import argparse
import json
from pathlib import Path
import torch

from src.models import RecSysNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Напр. models/recsys_nn_v1")
    parser.add_argument("--out_dir", type=str, required=True, help="Напр. torchserve_artifacts/recsys_nn_v1")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Не найден файл весов: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Не найден config.json: {config_path}. "
            f"Он должен содержать n_users/n_items/embedding_dim/hidden_dim/dropout"
        )

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = RecSysNN(**cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    state_out = out_dir / "model_state_dict.pt"
    torch.save(model.state_dict(), state_out)

    example_users = torch.zeros(1, dtype=torch.long)
    example_items = torch.zeros(1, dtype=torch.long)
    scripted = torch.jit.trace(model, (example_users, example_items))
    scripted.save(str(out_dir / "model.pt"))

    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: saved {state_out} and {out_dir / 'model.pt'} and {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
