import argparse
import os
import json
import pdfplumber
import yaml


def extract_tables(pdf_path: str):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                # Normalize rows
                rows = []
                for row in table:
                    if row is None:
                        continue
                    rows.append([c.strip() if isinstance(c, str) else c for c in row])
                if rows:
                    tables.append(rows)
    return tables


def guess_table_by_keywords(tables, keywords):
    for rows in tables:
        header = " ".join([str(x) for x in rows[0] if x])[:300].lower()
        if all(k.lower() in header for k in keywords):
            return rows
    return None


def update_configs(features, hyperparams):
    # Update uk/us config selected_features and model/train params per Table 3
    for cfg_path in ["config/uk_config.yaml", "config/us_config.yaml"]:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if features:
                cfg["selected_features"] = features
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
    model_cfg_path = "config/cnn_bilstm_attn.yaml"
    if os.path.exists(model_cfg_path) and hyperparams:
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            mcfg = yaml.safe_load(f)
        # Minimal mapping from hyperparams dict
        mcfg["data"]["batch_size"] = int(hyperparams.get("batch_size", mcfg["data"]["batch_size"]))
        mcfg["train"]["lr"] = float(hyperparams.get("learning_rate", mcfg["train"]["lr"]))
        mcfg["train"]["epochs"] = int(hyperparams.get("epochs", mcfg["train"]["epochs"]))
        mcfg["model"]["cnn_channels"][0] = int(hyperparams.get("cnn_channels_1", mcfg["model"]["cnn_channels"][0]))
        mcfg["model"]["cnn_channels"][1] = int(hyperparams.get("cnn_channels_2", mcfg["model"]["cnn_channels"][1]))
        mcfg["model"]["lstm_hidden"] = int(hyperparams.get("lstm_hidden", mcfg["model"]["lstm_hidden"]))
        with open(model_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(mcfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/papers/paper.pdf")
    parser.add_argument("--out", default="outputs/pdf_extract.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tables = extract_tables(args.pdf)

    # Heuristic identification
    table1 = guess_table_by_keywords(tables, ["indicator", "feature"]) or guess_table_by_keywords(tables, ["variable"])
    table3 = guess_table_by_keywords(tables, ["hyperparameter"]) or guess_table_by_keywords(tables, ["parameter", "value"])

    extracted = {"table1": table1, "table3": table3}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    # Parse minimal structures
    features = []
    if table1:
        # assume first column holds feature names (skip header)
        for row in table1[1:]:
            if row and row[0]:
                features.append(str(row[0]).strip())
    hyperparams = {}
    if table3:
        # assume two-column key/value
        for row in table3[1:]:
            if row and len(row) >= 2 and row[0] and row[1]:
                key = str(row[0]).strip().lower().replace(" ", "_")
                val = str(row[1]).strip()
                hyperparams[key] = val

    if features or hyperparams:
        update_configs(features, hyperparams)
        print("Configs updated based on extracted tables.")
    else:
        print("Could not confidently parse tables. Please provide screenshots of Table 1 and 3.")


if __name__ == "__main__":
    main()