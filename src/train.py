import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from tqdm import tqdm

from src.config import load_config
from src.model import CNNBiLSTMAttn, SimplifiedRiskModel
from src.metrics import compute_mae, classification_report_123
from src.plots import plot_metric_curves


# Time Series Data Augmentation Techniques
class TimeSeriesAugmenter:
    """Class for time series data augmentation"""
    
    @staticmethod
    def jitter(x, sigma=0.03):
        """Add random noise to each point"""
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    @staticmethod
    def scaling(x, sigma=0.1):
        """Apply random scaling"""
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], 1, x.shape[2]))
        return x * factor
    
    @staticmethod
    def time_warp(x, sigma=0.2, knot=4):
        """Apply random time warping"""
        orig_steps = np.arange(x.shape[1])
        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
        warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1]-1., num=knot+2))).T
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
                scale = (x.shape[1]-1)/time_warp[-1]
                ret[i, :, dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:, dim]).T
        return ret
    
    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        """Slice a random window from the time series"""
        target_len = int(reduce_ratio * x.shape[1])
        if target_len >= x.shape[1]:
            return x
        
        starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
        ends = (starts + target_len).astype(int)
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), 
                                          np.arange(target_len), 
                                          pat[starts[i]:ends[i], dim]).T
        return ret
    
    @staticmethod
    def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
        """Warp a window of the time series by speeding it up or down"""
        warp_scales = np.random.choice(scales, x.shape[0])
        warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)
        
        window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)
            
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[:window_starts[i], dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), 
                                     window_steps, pat[window_starts[i]:window_ends[i], dim])
                end_seg = pat[window_ends[i]:, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                ret[i, :, dim] = np.interp(np.arange(x.shape[1]), 
                                         np.linspace(0, warped.size-1, num=warped.size), 
                                         warped).T
        return ret
    
    @staticmethod
    def apply_augmentation(x, y, config):
        """Apply random augmentation based on config"""
        if not config.get('use_augmentation', False):
            return x, y
            
        # Make a copy to avoid modifying original data
        x_aug = x.copy()
        
        # Apply augmentations with probability
        if config.get('jitter', False) and np.random.random() < 0.5:
            x_aug = TimeSeriesAugmenter.jitter(x_aug, sigma=config.get('jitter_sigma', 0.03))
            
        if config.get('scaling', False) and np.random.random() < 0.5:
            x_aug = TimeSeriesAugmenter.scaling(x_aug, sigma=config.get('scaling_sigma', 0.1))
            
        if config.get('window_slice', False) and np.random.random() < 0.5:
            x_aug = TimeSeriesAugmenter.window_slice(x_aug, reduce_ratio=config.get('slice_ratio', 0.9))
            
        if config.get('window_warp', False) and np.random.random() < 0.5:
            x_aug = TimeSeriesAugmenter.window_warp(x_aug, 
                                                  window_ratio=config.get('warp_ratio', 0.1),
                                                  scales=config.get('warp_scales', [0.5, 2.]))
        
        return x_aug, y


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str, seq_len: int):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        window = self.df.iloc[idx: idx + self.seq_len]
        # Convert all columns to numeric, handling any non-numeric values
        numeric_data = window[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        x = numeric_data.values.astype(np.float32)
        y = window[self.target_col].values[-1].astype(np.float32)
        return torch.from_numpy(x), torch.tensor(y)


class FlatWindowDataset(Dataset):
    # Flattened windows for resampling
    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str, seq_len: int):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        window = self.df.iloc[idx: idx + self.seq_len]
        # Convert all columns to numeric, handling any non-numeric values
        numeric_data = window[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        x = numeric_data.values.astype(np.float32).reshape(-1)
        y = window[self.target_col].values[-1].astype(np.float32)
        return x, y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_mse_loss(pred, target, weights_map):
    target_cls = torch.clamp(target.round().long(), 1, 3)
    weights = torch.tensor([weights_map[1], weights_map[2], weights_map[3]], device=pred.device)
    w = weights[target_cls - 1]
    return torch.mean(w * (pred - target) ** 2)


def resample_windows(df: pd.DataFrame, feature_cols, target_col: str, seq_len: int, smote_cfg: dict, under_cfg: dict):
    ds = FlatWindowDataset(df, feature_cols, target_col, seq_len)
    X = []
    y = []
    for i in range(len(ds)):
        xi, yi = ds[i]
        X.append(xi)
        y.append(yi)
    X = np.stack(X)
    y = np.array(y).round().astype(int)  # class labels 1/2/3 for resampling

    steps = []
    if smote_cfg is not None:
        steps.append(("smote", SMOTE(**smote_cfg)))
    if under_cfg is not None:
        steps.append(("under", RandomUnderSampler(**under_cfg)))
    
    if not steps:
        return X.reshape(-1, seq_len, len(feature_cols)).astype(np.float32), y.astype(np.float32)
    
    pipe = ImbPipeline(steps=steps)
    X_rs, y_rs = pipe.fit_resample(X, y)

    # reshape back to (N, seq_len, features)
    features = len(feature_cols)
    X_rs_seq = X_rs.reshape(-1, seq_len, features).astype(np.float32)
    y_rs_seq = y_rs.astype(np.float32)
    return X_rs_seq, y_rs_seq


def filter_topk_features(feature_cols, shap_csv_path: str, topk: int):
    if topk is None or topk <= 0:
        return feature_cols
    if not os.path.exists(shap_csv_path):
        print(f"SHAP ranking file not found: {shap_csv_path}. Using all features.")
        return feature_cols
    rank_df = pd.read_csv(shap_csv_path)
    top_feats = rank_df["feature"].tolist()[:topk]
    filtered = [f for f in feature_cols if f in top_feats]
    if len(filtered) == 0:
        print("No overlap between SHAP top-k and available features. Using all features.")
        return feature_cols
    print(f"Using top-{topk} features ({len(filtered)} overlapped).")
    return filtered


def train_eval(model, opt, device, train_loader, val_loader, cfg, override_epochs: int = None):
    weights_map = cfg["train"]["sample_weighting"]
    history = []
    best_val = float("inf")
    patience = cfg["train"]["early_stopping_patience"]
    patience_ctr = 0
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=patience//2,
        min_lr=1e-6
    )
    
    # Add weight decay if model has it defined
    weight_decay = getattr(model, 'weight_decay', 0.0)
    
    # Add mixup augmentation parameters
    use_mixup = cfg.get("train", {}).get("use_mixup", True)
    mixup_alpha = cfg.get("train", {}).get("mixup_alpha", 0.2)
    
    def mixup_data(x, y, alpha=0.2):
        """Applies mixup augmentation to the batch"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(loss_fn, pred, y_a, y_b, lam):
        """Applies mixup to the loss calculation"""
        return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)
    
    def run_epoch(loader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        losses = []
        preds, trues = [], []
        
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Apply mixup augmentation during training
            if train_mode and use_mixup and np.random.random() < 0.7:  # 70% chance to apply mixup
                xb, yb_a, yb_b, lam = mixup_data(xb, yb, alpha=mixup_alpha)
                mixup_applied = True
            else:
                mixup_applied = False
            
            if train_mode:
                opt.zero_grad()
                
            with torch.set_grad_enabled(train_mode):
                yp = model(xb)
                
                # Apply L2 regularization manually if needed
                l2_reg = 0.0
                if weight_decay > 0:
                    for param in model.parameters():
                        l2_reg += torch.norm(param, 2)
                
                # Calculate loss with or without mixup
                if train_mode and mixup_applied:
                    def loss_fn(pred, target):
                        return weighted_mse_loss(pred, target, weights_map)
                    
                    loss = mixup_criterion(loss_fn, yp, yb_a, yb_b, lam)
                else:
                    loss = weighted_mse_loss(yp, yb, weights_map)
                
                # Add L2 regularization to loss
                if weight_decay > 0:
                    loss = loss + weight_decay * l2_reg
                
                if train_mode:
                    loss.backward()
                    if cfg["train"]["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"]) 
                    opt.step()
                    
            losses.append(loss.item())
            preds.append(yp.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
            
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mae = compute_mae(trues, preds)
        p, r, f1 = classification_report_123(trues, preds)
        return float(np.mean(losses)), mae, p, r, f1

    epochs = override_epochs if override_epochs and override_epochs > 0 else cfg["train"]["epochs"]
    best_epoch = 0
    min_epochs = cfg.get("train", {}).get("min_epochs", 10)  # Minimum epochs before early stopping
    
    for ep in range(1, epochs + 1):
        tr_loss, tr_mae, tr_p, tr_r, tr_f1 = run_epoch(train_loader, True)
        va_loss, va_mae, va_p, va_r, va_f1 = run_epoch(val_loader, False)
        
        # Update learning rate scheduler
        scheduler.step(va_mae)
        
        # Log metrics
        print(f"Epoch {ep}/{epochs} - Train: loss={tr_loss:.4f}, mae={tr_mae:.4f} | Val: loss={va_loss:.4f}, mae={va_mae:.4f}, P={va_p:.4f}, R={va_r:.4f}, F1={va_f1:.4f}")
        
        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_mae": tr_mae,
            "val_loss": va_loss,
            "val_mae": va_mae,
            "val_p": va_p,
            "val_r": va_r,
            "val_f1": va_f1,
        })
        
        # Early stopping with minimum epochs requirement
        if va_mae < best_val:
            best_val = va_mae
            best_epoch = ep
            patience_ctr = 0
            print(f"✓ New best model with MAE: {va_mae:.4f}")
        else:
            patience_ctr += 1
            # Only consider early stopping after minimum epochs
            if ep >= min_epochs and patience_ctr >= patience:
                print(f"Early stopping at epoch {ep}. Best was epoch {best_epoch} with MAE: {best_val:.4f}")
                break
    return history, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--topk", type=int, default=0, help="Use top-k features from SHAP ranking if >0")
    parser.add_argument("--shap_ranking", type=str, default="outputs/shap_global_ranking.csv")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs from config if >0")
    parser.add_argument("--tag", type=str, default="", help="Write outputs under outputs/{tag}/ if provided")
    parser.add_argument("--use_simplified_model", action="store_true", help="Use simplified model for small datasets")
    args = parser.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"]) if "seed" in cfg else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    target_col = "risk_value"

    exclude = {target_col}
    feature_cols_all = [c for c in train_df.columns if c not in exclude]

    feature_cols = filter_topk_features(feature_cols_all, args.shap_ranking, args.topk)

    seq_len = cfg["data"]["sequence_length"]
    batch_size = cfg["data"]["batch_size"]

    X_rs, y_rs = resample_windows(
        train_df,
        feature_cols,
        target_col,
        seq_len,
        cfg["balance"]["smote"],
        cfg["balance"]["undersample"],
    )

    X_rs_t = torch.from_numpy(X_rs)
    y_rs_t = torch.from_numpy(y_rs)

    class TensorSeqDataset(Dataset):
        def __len__(self):
            return X_rs_t.shape[0]
        def __getitem__(self, idx):
            return X_rs_t[idx], y_rs_t[idx]

    rs_train_ds = TensorSeqDataset()

    val_ds = SeqDataset(val_df, feature_cols, target_col, seq_len)
    test_ds = SeqDataset(test_df, feature_cols, target_col, seq_len)

    in_channels = X_rs.shape[2]

    if cfg.get("cv", {}).get("folds", 0) and cfg.get("train", {}).get("use_cv", False):
        k = int(cfg["cv"]["folds"]) or 10
        kf = KFold(n_splits=k, shuffle=True, random_state=cfg.get("seed", 42))
        fold_maes = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(rs_train_ds))), 1):
            tr_loader = DataLoader(Subset(rs_train_ds, tr_idx), batch_size=batch_size, shuffle=True, num_workers=cfg["data"]["num_workers"]) 
            va_loader = DataLoader(Subset(rs_train_ds, va_idx), batch_size=batch_size, shuffle=False, num_workers=cfg["data"]["num_workers"]) 

            model_cv = CNNBiLSTMAttn(
                in_channels=in_channels,
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
            opt_cv = torch.optim.Adam(model_cv.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]) 
            _, best_va = train_eval(model_cv, opt_cv, device, tr_loader, va_loader, cfg, override_epochs=args.epochs)
            fold_maes.append(best_va)
            print(f"CV fold {fold}/{k}: best val MAE={best_va:.4f}")
        print(f"CV mean val MAE={np.mean(fold_maes):.4f} +/- {np.std(fold_maes):.4f}")

    train_loader = DataLoader(rs_train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg["data"]["num_workers"]) 
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg["data"]["num_workers"]) 
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=cfg["data"]["num_workers"]) 

    if args.use_simplified_model:
        model = SimplifiedRiskModel(
            in_channels=in_channels,
            hidden_dim=cfg["model"]["fc_dim"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
    else:
        model = CNNBiLSTMAttn(
            in_channels=in_channels,
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

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]) 

    # Determine output dir with optional tag
    base_out = cfg["output"]["dir"]
    out_dir = os.path.join(base_out, args.tag) if args.tag else base_out
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, cfg["output"]["checkpoint"])
    log_csv = os.path.join(out_dir, "metrics_log.csv")

    history, _ = train_eval(model, opt, device, train_loader, val_loader, cfg, override_epochs=args.epochs)

    torch.save({"model": model.state_dict(), "cfg": cfg, "features": feature_cols}, ckpt_path)
    pd.DataFrame(history).to_csv(log_csv, index=False)
    plot_metric_curves(history, os.path.join(out_dir, "mae_curves.png"))

    def evaluate(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yp = model(xb)
                preds.append(yp.detach().cpu().numpy())
                trues.append(yb.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mae = compute_mae(trues, preds)
        p, r, f1 = classification_report_123(trues, preds)
        return mae, p, r, f1

    te_mae, te_p, te_r, te_f1 = evaluate(test_loader)
    print(f"TEST: MAE={te_mae:.4f} P={te_p:.4f} R={te_r:.4f} F1={te_f1:.4f}")

    final_row = pd.DataFrame([{"epoch": len(history), "test_mae": te_mae, "test_p": te_p, "test_r": te_r, "test_f1": te_f1}])
    final_row.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)


if __name__ == "__main__":
    main()