#!/usr/bin/env python3
"""
Dataset preprocessing script that matches the IEEE paper's exact settings.
Handles both UK and US datasets with proper SMOTE+UnderSampler configuration.

Usage:
  python scripts/dataset_preprocessing.py --dataset UK --config config/uk_config.yaml
  python scripts/dataset_preprocessing.py --dataset US --config config/us_config.yaml
"""
import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def verify_paper_settings(config, dataset_name):
    """Verify that config matches paper settings"""
    print(f"🔍 Verifying {dataset_name} dataset configuration...")
    
    expected_settings = {
        'missing_threshold': 0.10,
        'near_zero_var_threshold': 0.02,
        'split': {
            'ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2}
        }
    }
    
    issues = []
    
    # Check missing threshold
    if config.get('missing_threshold') != expected_settings['missing_threshold']:
        issues.append(f"missing_threshold should be {expected_settings['missing_threshold']}")
    
    # Check near zero variance threshold
    if config.get('near_zero_var_threshold') != expected_settings['near_zero_var_threshold']:
        issues.append(f"near_zero_var_threshold should be {expected_settings['near_zero_var_threshold']}")
    
    # Check split ratios
    ratios = config.get('split', {}).get('ratios', {})
    expected_ratios = expected_settings['split']['ratios']
    for key, expected_val in expected_ratios.items():
        if ratios.get(key) != expected_val:
            issues.append(f"split ratio {key} should be {expected_val}")
    
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Configuration matches paper settings")
        return True


def check_uk_features(config):
    """Check UK dataset has the 32 features from Table 1"""
    expected_features = [
        # Temporal (4)
        'year', 'month', 'Day_of_Week', 'hour',
        # Spatial (7)
        'Latitude', 'Longitude', 'Junction_Control', 'Junction_Detail', 
        'Junction_Location', 'Number_of_Vehicles', 'Pedestrian_Crossing-Physical_Facilities',
        # Environmental (8)
        'Light_Conditions', 'Weather_Conditions', 'Urban_or_Rural_Area',
        'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Road_Type',
        'Road_Surface_Conditions', 'Speed_limit',
        # Vehicle factors (8)
        'Propulsion_Code', 'Age_of_Vehicle', 'Engine_Capacity_(CC)',
        'Towing_and_Articulation', 'Skidding_and_Overturning', 'Vehicle_Manoeuvre',
        'Vehicle_Type', 'Vehicle_Reference',
        # Driver factors (5)
        'Age_Band_of_Driver', 'Driver_Home_Area_Type', 'Journey_Purpose_of_Driver',
        'Sex_of_Driver', 'Was_Vehicle_Left_Hand_Drive'
    ]
    
    config_features = config.get('selected_features', [])
    
    print(f"📊 UK Features check:")
    print(f"   Expected: {len(expected_features)} features")
    print(f"   Config: {len(config_features)} features")
    
    missing = set(expected_features) - set(config_features)
    extra = set(config_features) - set(expected_features)
    
    if missing:
        print(f"   ⚠️  Missing features: {missing}")
    if extra:
        print(f"   ℹ️  Extra features: {extra}")
    
    if not missing:
        print("   ✅ All expected features present")
        return True
    return False


def create_smote_config():
    """Create SMOTE+UnderSampler config matching the paper"""
    return {
        'smote': {
            'random_state': 42,
            'k_neighbors': 5,
            'sampling_strategy': 'auto'
        },
        'undersample': {
            'random_state': 42,
            'sampling_strategy': 'auto'
        }
    }


def run_preprocessing(config_path, dataset_name):
    """Run the preprocessing with paper-compliant settings"""
    print(f"\n🔄 Running preprocessing for {dataset_name} dataset...")
    
    # Load and verify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify settings
    if not verify_paper_settings(config, dataset_name):
        print("❌ Config verification failed")
        return False
    
    if dataset_name == 'UK':
        if not check_uk_features(config):
            print("⚠️  UK features check failed - continuing anyway")
    
    # Update config with proper SMOTE settings
    smote_config = create_smote_config()
    config['balance'] = smote_config
    
    # Save updated config
    updated_config_path = config_path.replace('.yaml', '_updated.yaml')
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"💾 Updated config saved to: {updated_config_path}")
    
    # Run preprocessing
    import subprocess
    cmd = [
        sys.executable, "-m", "src.preprocess",
        "--config", updated_config_path
    ]
    
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Preprocessing completed successfully")
        print(result.stdout)
        return True
    else:
        print("❌ Preprocessing failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def analyze_processed_data(processed_dir):
    """Analyze the processed data to verify it matches paper expectations"""
    print(f"\n📊 Analyzing processed data in {processed_dir}...")
    
    try:
        train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
        
        print(f"📈 Dataset sizes:")
        print(f"   Train: {len(train_df):,} samples")
        print(f"   Val: {len(val_df):,} samples")
        print(f"   Test: {len(test_df):,} samples")
        
        # Check risk value distribution
        if 'risk_value' in train_df.columns:
            risk_dist = train_df['risk_value'].value_counts().sort_index()
            print(f"📊 Risk value distribution (train):")
            for risk, count in risk_dist.items():
                pct = count / len(train_df) * 100
                print(f"   Risk {risk}: {count:,} ({pct:.1f}%)")
        
        # Check feature count
        feature_cols = [c for c in train_df.columns if c != 'risk_value']
        print(f"🔢 Features: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['UK', 'US'], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing processed data")
    args = parser.parse_args()
    
    print(f"🚀 Dataset preprocessing for {args.dataset}")
    print(f"📁 Config: {args.config}")
    
    if args.analyze_only:
        # Just analyze existing data
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        processed_dir = config.get('processed_dir', f'data/processed/{args.dataset.lower()}')
        analyze_processed_data(processed_dir)
    else:
        # Run full preprocessing
        success = run_preprocessing(args.config, args.dataset)
        
        if success:
            # Analyze results
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            processed_dir = config.get('processed_dir', f'data/processed/{args.dataset.lower()}')
            analyze_processed_data(processed_dir)
            
            print(f"\n🎉 {args.dataset} dataset preprocessing complete!")
        else:
            print(f"\n❌ {args.dataset} dataset preprocessing failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
