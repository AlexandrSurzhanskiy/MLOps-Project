import torch
import torch.nn as nn


class RecSysNN(nn.Module):
    def __init__(
        self, n_users, n_items, embedding_dim=32, hidden_dim=64, dropout=0.2
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze()

    def save_pretrained(self, output_dir):
        import os, json, torch

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), f"{output_dir}/pytorch_model.bin")
        cfg = {
            "n_users": self.user_emb.num_embeddings,
            "n_items": self.item_emb.num_embeddings,
            "embedding_dim": self.user_emb.embedding_dim,
        }
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(cfg, f, indent=2)
