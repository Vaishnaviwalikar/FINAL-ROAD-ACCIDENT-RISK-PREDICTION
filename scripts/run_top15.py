#!/usr/bin/env python3
"""
Automate the two-step pipeline described in the paper:
1) Generate DeepSHAP global ranking from a trained checkpoint
2) Retrain using the top-K features and evaluate

Usage examples:
  python scripts/run_top15.py --checkpoint outputs/uk_full_e100/best.pt --config config/uk_config.yaml --topk 15 --tag uk_top15_e50 --epochs 50

This script assumes your processed data and checkpoint were created via src/preprocess.py and src/train.py.
"""
import argparse
import os
import subprocess
import sys


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--tag", type=str, default="uk_top15")
    p.add_argument("--background", type=int, default=500)
    args = p.parse_args()

    # 1) Generate DeepSHAP global ranking
    run([
        sys.executable, "-m", "src.explain_deepshap",
        "--checkpoint", args.checkpoint,
        "--background-samples", str(args.background),
        "--tag", args.tag
    ])

    # 2) Retrain using top-k features from the generated ranking
    shap_rank_path = os.path.join("outputs", args.tag, "shap_global_ranking.csv")
    run([
        sys.executable, "-m", "src.train",
        "--config", args.config,
        "--topk", str(args.topk),
        "--shap_ranking", shap_rank_path,
        "--epochs", str(args.epochs),
        "--tag", args.tag
    ])


if __name__ == "__main__":
    main()
