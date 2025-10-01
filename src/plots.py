import os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_curves(history: List[Dict[str, float]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = np.arange(1, len(history) + 1)
    df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_mae"], label="train_mae")
    plt.plot(epochs, df["val_mae"], label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distribution(df: pd.DataFrame, col: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], bins=30, kde=False)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()