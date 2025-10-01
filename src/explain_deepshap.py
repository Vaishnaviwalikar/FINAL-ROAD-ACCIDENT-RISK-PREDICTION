import argparse
import os
import numpy as np
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader, Dataset

from config import load_config
from model import CNNBiLSTMAttn


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, seq_len: int):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        window = self.df.iloc[idx: idx + self.seq_len]
        x = window[self.feature_cols].values.astype(np.float32)
        return torch.from_numpy(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--background-samples", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]

    base_model = CNNBiLSTMAttn(
        in_channels=cfg["model"]["in_channels"],
        cnn_channels=tuple(cfg["model"]["cnn_channels"]),
        kernel_sizes=tuple(cfg["model"]["kernel_sizes"]),
        pool_size=cfg["model"]["pool_size"],
        fc_dim=cfg["model"]["fc_dim"],
        attn_spatial_dim=cfg["model"]["attn_spatial_dim"],
        attn_temporal_dim=cfg["model"]["attn_temporal_dim"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    base_model.load_state_dict(ckpt["model"]) 
    base_model.eval()
    
    # Wrapper for DeepSHAP (needs 2D output)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            return self.base_model(x).unsqueeze(-1)  # Add dimension for DeepSHAP
    
    model = ModelWrapper(base_model)

    data_dir = cfg["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    feature_cols = ckpt["features"]

    seq_len = cfg["data"]["sequence_length"]

    # Background (subset of training)
    bg = train_df.sample(n=min(args.background_samples, len(train_df)), random_state=cfg["seed"]) 
    bg_ds = SeqDataset(bg, feature_cols, seq_len)
    bg_batch = torch.stack([bg_ds[i] for i in range(len(bg_ds))], dim=0).to(device)

    explainer = shap.DeepExplainer(model, bg_batch)

    # Evaluate on a held-out subset of train
    X = train_df.sample(n=min(2000, len(train_df)), random_state=cfg["seed"]) 
    X_ds = SeqDataset(X, feature_cols, seq_len)

    shap_sums = np.zeros(len(feature_cols), dtype=np.float64)
    count = 0

    for i in range(0, len(X_ds), args.batch):
        batch = torch.stack([X_ds[j] for j in range(i, min(i + args.batch, len(X_ds)))], dim=0).to(device)
        sv = explainer.shap_values(batch)
        if isinstance(sv, torch.Tensor):
            sv = sv.detach().cpu().numpy()
        elif isinstance(sv, list) and isinstance(sv[0], torch.Tensor):
            sv = sv[0].detach().cpu().numpy()
        shap_sums += np.abs(sv).mean(axis=1).sum(axis=0)
        count += sv.shape[0]

    mean_abs = shap_sums / max(count, 1)
    ranking = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)

    base_out = cfg["output"]["dir"]
    out_dir = os.path.join(base_out, args.tag) if args.tag else base_out
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(ranking, columns=["feature", "mean_abs_shap"]).to_csv(os.path.join(out_dir, "shap_global_ranking.csv"), index=False)
    print(f"Saved SHAP global ranking -> {os.path.join(out_dir, 'shap_global_ranking.csv')} ")


if __name__ == "__main__":
    main()