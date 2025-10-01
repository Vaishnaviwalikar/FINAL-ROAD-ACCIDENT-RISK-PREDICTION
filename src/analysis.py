import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_counts(df: pd.DataFrame, col: str, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    cnt = df[col].value_counts().sort_index()
    sns.barplot(x=cnt.index, y=cnt.values, color="#4C72B0")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    train = pd.read_csv(os.path.join(args.processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.processed_dir, "test.csv"))

    df = pd.concat([train, val, test], axis=0, ignore_index=True)

    if "year" in df.columns:
        plot_counts(df, "year", os.path.join(args.out, "annual_counts.png"), "Annual accident counts")
    if "week" in df.columns:
        plot_counts(df, "week", os.path.join(args.out, "weekly_counts.png"), "Weekly accident counts")
    if "hour" in df.columns:
        plot_counts(df, "hour", os.path.join(args.out, "hourly_counts.png"), "Hourly accident counts")

    print(f"Saved analysis plots to {args.out}")


if __name__ == "__main__":
    main()