import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from config import load_config
from model import CNNBiLSTMAttn


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str, seq_len: int):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.df) - self.seq_len + 1
    
    def __getitem__(self, idx):
        window = self.df.iloc[idx:idx + self.seq_len]
        x = window[self.feature_cols].values.astype(np.float32)
        y = window[self.target_col].iloc[-1]  # Use last value in sequence as target
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)


def compute_permutation_importance(model, data_loader, feature_cols, device, n_repeats=5):
    """Compute permutation importance for each feature."""
    model.eval()
    
    # Get baseline predictions
    baseline_preds = []
    baseline_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            baseline_preds.extend(pred.cpu().numpy())
            baseline_targets.extend(y.cpu().numpy())
    
    baseline_preds = np.array(baseline_preds)
    baseline_targets = np.array(baseline_targets)
    baseline_mae = mean_absolute_error(baseline_targets, baseline_preds)
    
    # Compute importance for each feature
    importance_scores = []
    
    for feat_idx in tqdm(range(len(feature_cols)), desc="Computing feature importance"):
        feature_scores = []
        
        for _ in range(n_repeats):
            # Create modified data with shuffled feature
            modified_preds = []
            
            with torch.no_grad():
                for batch in data_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    
                    # Shuffle the specific feature across the batch
                    x_modified = x.clone()
                    x_modified[:, :, feat_idx] = x_modified[:, :, feat_idx][torch.randperm(x.size(0))]
                    
                    pred = model(x_modified)
                    modified_preds.extend(pred.cpu().numpy())
            
            modified_preds = np.array(modified_preds)
            modified_mae = mean_absolute_error(baseline_targets, modified_preds)
            feature_scores.append(modified_mae - baseline_mae)
        
        importance_scores.append(np.mean(feature_scores))
    
    return importance_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tag", required=True, help="Tag for output files")
    parser.add_argument("--n-repeats", type=int, default=3, help="Number of repeats for permutation")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    
    model = CNNBiLSTMAttn(
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
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Load data
    data_dir = cfg["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    feature_cols = ckpt["features"]
    seq_len = cfg["data"]["sequence_length"]
    
    # Create dataset and dataloader
    target_col = "risk_value"
    dataset = SeqDataset(train_df, feature_cols, target_col, seq_len)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Compute importance
    print("Computing permutation importance...")
    importance_scores = compute_permutation_importance(
        model, data_loader, feature_cols, device, args.n_repeats
    )
    
    # Create ranking
    feature_ranking = list(zip(feature_cols, importance_scores))
    feature_ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Save results
    output_dir = f"outputs/{args.tag}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ranking
    ranking_df = pd.DataFrame(feature_ranking, columns=["feature", "importance"])
    ranking_path = os.path.join(output_dir, "feature_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    
    print(f"\nTop 15 most important features:")
    for i, (feat, score) in enumerate(feature_ranking[:15]):
        print(f"{i+1:2d}. {feat:<30} {score:.4f}")
    
    print(f"\nFeature ranking saved to: {ranking_path}")
    
    # Save top-15 features for retraining
    top15_features = [feat for feat, _ in feature_ranking[:15]]
    top15_path = os.path.join(output_dir, "top15_features.txt")
    with open(top15_path, "w") as f:
        for feat in top15_features:
            f.write(f"{feat}\n")
    
    print(f"Top-15 features list saved to: {top15_path}")


if __name__ == "__main__":
    main()