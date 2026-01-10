import json
from typing import Any, List
import torch
from ts.torch_handler.base_handler import BaseHandler


class RecSysTorchScriptHandler(BaseHandler):
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties

        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.jit.load(f"{model_dir}/model.pt", map_location=self.device)
        self.model.eval()

    def preprocess(self, data: List[dict]) -> Any:
        req = data[0]
        raw = req.get("body") or req.get("data")

        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")

        if isinstance(raw, str):
            payload = json.loads(raw)
        elif isinstance(raw, dict):
            payload = raw
        else:
            raise ValueError(f"Unsupported request type: {type(raw)}")

        users = payload["user_idx"]
        items = payload["item_idx"]

        if len(users) != len(items):
            raise ValueError("user_idx и item_idx должны быть одинаковой длины")

        users_t = torch.tensor(users, dtype=torch.long, device=self.device)
        items_t = torch.tensor(items, dtype=torch.long, device=self.device)
        return users_t, items_t

    def inference(self, inputs, *args, **kwargs):
        users_t, items_t = inputs
        with torch.no_grad():
            preds = self.model(users_t, items_t).view(-1)
        return preds.detach().cpu().numpy().tolist()

    def postprocess(self, preds):
        return [json.dumps({"predicted_score": preds}, ensure_ascii=False)]
