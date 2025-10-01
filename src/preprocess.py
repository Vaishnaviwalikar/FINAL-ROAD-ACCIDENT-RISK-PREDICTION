import argparse
import os
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from joblib import dump
from .config import load_config


def remove_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    keep = [c for c in df.columns if df[c].isna().mean() <= threshold]
    return df[keep]


def remove_near_zero_variance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # Heuristic: keep columns where top category proportion < 1 - threshold
    keep_cols = []
    for c in df.columns:
        vc = df[c].value_counts(normalize=True, dropna=False)
        if vc.empty:
            continue
        if vc.iloc[0] < (1 - threshold):
            keep_cols.append(c)
    return df[keep_cols]


def impute_and_label_encode(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, dict]:
    label_encoders = {}
    for c in categorical_cols:
        mode_val = df[c].mode(dropna=True)
        if len(mode_val) > 0:
            df[c] = df[c].fillna(mode_val.iloc[0])
        else:
            df[c] = df[c].fillna("UNKNOWN")
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le
    return df, label_encoders


def add_time_parts(df: pd.DataFrame, date_col: str, time_col: str = None) -> pd.DataFrame:
    if date_col in df.columns:
        if time_col and time_col in df.columns:
            dt = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors="coerce")
        else:
            dt = pd.to_datetime(df[date_col], errors="coerce")
        df["timestamp"] = dt
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["week"] = dt.dt.isocalendar().week.astype(int)
        df["hour"] = dt.dt.hour.fillna(0).astype(int)
    return df


def build_segment_id(df: pd.DataFrame, segment_id_col: str) -> pd.DataFrame:
    if segment_id_col in df.columns:
        return df
    # Fallback: build from lat/lon grid (0.01 deg buckets)
    lat_cols = [c for c in df.columns if c.lower() in ["latitude", "lat"]]
    lon_cols = [c for c in df.columns if c.lower() in ["longitude", "lon", "lng"]]
    if lat_cols and lon_cols:
        lat_c = lat_cols[0]
        lon_c = lon_cols[0]
        latq = (df[lat_c].astype(float) / 0.01).round().astype('Int64')
        lonq = (df[lon_c].astype(float) / 0.01).round().astype('Int64')
        df["Road_Segment_Id"] = latq.astype(str) + ":" + lonq.astype(str)
    else:
        df["Road_Segment_Id"] = "SEG_UNKNOWN"
    return df


def construct_target(df: pd.DataFrame, dataset_name: str, segment_id_col: str, period_unit: str) -> pd.DataFrame:
    df = df.copy()
    # Map severity to 1/2/3
    if dataset_name == "US":
        if "Severity" in df.columns:
            df["risk"] = df["Severity"].map({1:1, 2:1, 3:2, 4:3}).fillna(1).astype(int)
        else:
            df["risk"] = 1
    else:
        if "Severity" in df.columns and df["Severity"].dtype == object:
            severity_map = {"Slight": 1, "Serious": 2, "Fatal": 3}
            df["risk"] = df["Severity"].map(severity_map).fillna(1).astype(int)
        elif "Accident_Severity" in df.columns:
            df["risk"] = df["Accident_Severity"].map({1:3, 2:2, 3:1}).fillna(1).astype(int)
        else:
            df["risk"] = 1

    if period_unit == "week":
        period_key = ["year", "week"]
    elif period_unit == "month":
        period_key = ["year", "month"]
    else:
        period_key = ["year"]

    group_cols = [segment_id_col] + [k for k in period_key if k in df.columns]
    agg = df.groupby(group_cols)["risk"].sum().reset_index().rename(columns={"risk": "risk_value"})
    df = df.merge(agg, on=group_cols, how="left")
    df["risk_value"] = df["risk_value"].clip(lower=1, upper=3)
    return df


def chronological_split(df: pd.DataFrame, ratios: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Prefer exact timestamp sorting if available
    sort_cols = [c for c in ["timestamp", "year", "month", "week", "hour"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols, axis=0, kind="mergesort")
    n = len(df_sorted)
    n_train = int(ratios["train"] * n)
    n_val = int(ratios["val"] * n)
    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train:n_train + n_val]
    test = df_sorted.iloc[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--row-limit", type=int, default=0, help="Limit rows read from each raw CSV")
    parser.add_argument("--sample-frac", type=float, default=0.0, help="Randomly sample this fraction after merge (0-1)")
    parser.add_argument("--accidents-only", action="store_true", help="Skip vehicle merge; use accidents table only")
    args = parser.parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["processed_dir"], exist_ok=True)

    nrows = args.row_limit if args.row_limit and args.row_limit > 0 else None

    if cfg.get("dataset") == "UK":
        left_path = os.path.join(cfg["raw_dir"], cfg.get("merge", {}).get("left", "Accidents.csv"))
        right_name = cfg.get("merge", {}).get("right")
        right_path = os.path.join(cfg["raw_dir"], right_name) if right_name else None
        on_key = cfg.get("merge", {}).get("on", "Accident_Index")
        left = pd.read_csv(left_path, nrows=nrows)
        if not args.accidents_only and right_path and os.path.exists(right_path):
            right = pd.read_csv(right_path, nrows=nrows)
            if on_key not in left.columns or on_key not in right.columns:
                for candidate in ["Accident_Index", "Accident Reference", "Accident_Ref", "ACCIDENT_INDEX"]:
                    if candidate in left.columns and candidate in right.columns:
                        on_key = candidate
                        break
            df = left.merge(right, on=on_key, how="left")
        else:
            df = left
        date_col = cfg["datetime_columns"]["accident_date"]
        time_col = cfg["datetime_columns"].get("accident_time")
    else:
        df = pd.read_csv(os.path.join(cfg["raw_dir"], "US_Accidents_March23.csv"), nrows=nrows)
        date_col = cfg["datetime_columns"]["accident_date"]
        time_col = None

    if 0.0 < args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=cfg.get("seed", 42))

    df = remove_high_missing(df, cfg["missing_threshold"])
    df = add_time_parts(df, date_col, time_col)
    df = build_segment_id(df, cfg["segment_id_col"])

    categorical_cols = [c for c in df.columns if df[c].dtype == object]
    df, label_encoders = impute_and_label_encode(df, categorical_cols)

    df = remove_near_zero_variance(df, cfg["near_zero_var_threshold"])

    selected_features = cfg.get("selected_features", [])
    cols = list(set(selected_features + [cfg["segment_id_col"], "year", "month", "week", "hour"]))
    cols = [c for c in cols if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]

    df = construct_target(df, cfg.get("dataset", "UK"), cfg["segment_id_col"], cfg["period"]["unit"])

    train_df, val_df, test_df = chronological_split(df, cfg["split"]["ratios"])

    out_dir = cfg["processed_dir"]
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Persist encoders
    dump(label_encoders, os.path.join(out_dir, "label_encoders.joblib"))

    # Compute and save a simple scaler (mean/std) on numeric feature columns of train only
    numeric_cols = [
        c for c in train_df.columns
        if c not in {"risk_value", "timestamp", "Date", "Time"}
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    train_numeric = train_df[numeric_cols].copy()
    means = train_numeric.mean().to_dict()
    stds = (train_numeric.std(ddof=0).replace(0, 1e-6)).to_dict()
    scaler_obj = {"features": numeric_cols, "mean": means, "std": stds}
    dump(scaler_obj, os.path.join(out_dir, "scaler.joblib"))

    print(f"Saved: {train_path}, {val_path}, {test_path}; total_rows={len(df)}")


if __name__ == "__main__":
    main()