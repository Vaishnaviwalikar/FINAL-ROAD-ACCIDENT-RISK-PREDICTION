import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", required=True, help="Path to features file")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    # Read top features
    with open(args.features_file, 'r') as f:
        top_features = [line.strip() for line in f if line.strip()]
    
    print(f"Selected {len(top_features)} features:")
    for i, feat in enumerate(top_features, 1):
        print(f"{i:2d}. {feat}")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = os.path.join(args.data_dir, f"{split}.csv")
        output_file = os.path.join(args.output_dir, f"{split}.csv")
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        df = pd.read_csv(input_file)
        
        # Keep only selected features plus target columns
        keep_cols = top_features + ['risk_value', 'risk']
        available_cols = [col for col in keep_cols if col in df.columns]
        
        if len(available_cols) < len(top_features):
            missing = set(top_features) - set(available_cols)
            print(f"Warning: Missing features in {split}: {missing}")
        
        df_filtered = df[available_cols]
        df_filtered.to_csv(output_file, index=False)
        print(f"Saved {len(df_filtered)} rows with {len(available_cols)} columns to {output_file}")

if __name__ == "__main__":
    main()