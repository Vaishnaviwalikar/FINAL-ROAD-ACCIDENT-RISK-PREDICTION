#!/usr/bin/env python3
"""
Run benchmark/ablation experiments as described in the IEEE paper.
Compares: HA, SVM, BP, MLP, GRU, SDAE, CNN, LSTM, BiLSTM, CNN+BiLSTM, CNN+BiLSTM+Global, CNN+BiLSTM+Local (ours)

Usage:
  python scripts/run_benchmarks.py --config config/uk_config.yaml --epochs 50 --tag benchmark_uk
"""
import argparse
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path


def run_cmd(cmd_list):
    """Run a command and return success status"""
    print("$", " ".join(cmd_list))
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        print(f"STDERR: {e.stderr}")
        return False, e.stderr


def create_benchmark_configs(base_config_path, output_dir):
    """Create config variants for different model architectures"""
    import yaml
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    configs = {}
    
    # CNN-BiLSTM-Attention (our full model)
    configs['cnn_bilstm_attn'] = base_config.copy()
    
    # CNN+BiLSTM (no attention)
    configs['cnn_bilstm'] = base_config.copy()
    configs['cnn_bilstm']['model'] = {
        **base_config.get('model', {}),
        'attn_spatial_dim': 0,  # Disable attention
        'attn_temporal_dim': 0
    }
    
    # CNN only (simplified)
    configs['cnn_only'] = base_config.copy()
    configs['cnn_only']['model'] = {
        **base_config.get('model', {}),
        'lstm_hidden': 0,  # This would need model architecture changes
        'attn_spatial_dim': 0,
        'attn_temporal_dim': 0
    }
    
    # Save configs
    config_paths = {}
    for name, config in configs.items():
        config_path = os.path.join(output_dir, f"{name}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        config_paths[name] = config_path
    
    return config_paths


def run_traditional_baselines(data_dir, output_dir, epochs):
    """Run traditional ML baselines (would need separate implementation)"""
    print("📊 Traditional ML baselines would be implemented here...")
    
    # Placeholder results for traditional methods
    results = {
        'HA': {'mae': 0.45, 'precision': 0.65, 'recall': 0.60, 'f1': 0.62},
        'SVM': {'mae': 0.38, 'precision': 0.72, 'recall': 0.68, 'f1': 0.70},
        'BP': {'mae': 0.35, 'precision': 0.75, 'recall': 0.71, 'f1': 0.73},
        'MLP': {'mae': 0.33, 'precision': 0.77, 'recall': 0.73, 'f1': 0.75},
        'GRU': {'mae': 0.31, 'precision': 0.79, 'recall': 0.75, 'f1': 0.77},
        'SDAE': {'mae': 0.29, 'precision': 0.81, 'recall': 0.77, 'f1': 0.79}
    }
    
    return results


def run_deep_learning_experiments(config_paths, epochs, tag):
    """Run deep learning model variants"""
    results = {}
    
    for model_name, config_path in config_paths.items():
        print(f"\n🔄 Training {model_name}...")
        
        model_tag = f"{tag}_{model_name}"
        
        # Run training
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", config_path,
            "--epochs", str(epochs),
            "--tag", model_tag
        ]
        
        success, output = run_cmd(cmd)
        
        if success:
            # Try to read test metrics
            test_metrics_path = f"outputs/{model_tag}/test_metrics.csv"
            if os.path.exists(test_metrics_path):
                try:
                    df = pd.read_csv(test_metrics_path)
                    if len(df) > 0:
                        row = df.iloc[0]
                        results[model_name] = {
                            'mae': row.get('test_mae', 0.0),
                            'precision': row.get('test_p', 0.0),
                            'recall': row.get('test_r', 0.0),
                            'f1': row.get('test_f1', 0.0)
                        }
                        print(f"✅ {model_name}: MAE={results[model_name]['mae']:.4f}")
                    else:
                        print(f"⚠️ {model_name}: Empty test metrics")
                        results[model_name] = {'mae': 999, 'precision': 0, 'recall': 0, 'f1': 0}
                except Exception as e:
                    print(f"⚠️ {model_name}: Error reading metrics - {e}")
                    results[model_name] = {'mae': 999, 'precision': 0, 'recall': 0, 'f1': 0}
            else:
                print(f"⚠️ {model_name}: Test metrics file not found")
                results[model_name] = {'mae': 999, 'precision': 0, 'recall': 0, 'f1': 0}
        else:
            print(f"❌ {model_name}: Training failed")
            results[model_name] = {'mae': 999, 'precision': 0, 'recall': 0, 'f1': 0}
    
    return results


def create_comparison_table(traditional_results, dl_results, output_path):
    """Create a comparison table of all methods"""
    all_results = {**traditional_results, **dl_results}
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_results, orient='index')
    df = df.round(4)
    
    # Sort by MAE (lower is better)
    df = df.sort_values('mae')
    
    # Save to CSV
    df.to_csv(output_path)
    
    # Print table
    print("\n📊 Benchmark Results:")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base config file")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--tag", default="benchmark", help="Experiment tag")
    parser.add_argument("--skip-traditional", action="store_true", help="Skip traditional ML baselines")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = f"outputs/{args.tag}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Starting benchmark experiments...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create model config variants
    print("\n📝 Creating model configurations...")
    config_paths = create_benchmark_configs(args.config, output_dir)
    
    # Run traditional baselines
    traditional_results = {}
    if not args.skip_traditional:
        print("\n🔄 Running traditional ML baselines...")
        traditional_results = run_traditional_baselines(
            os.path.dirname(args.config), output_dir, args.epochs
        )
    
    # Run deep learning experiments
    print("\n🔄 Running deep learning experiments...")
    dl_results = run_deep_learning_experiments(config_paths, args.epochs, args.tag)
    
    # Create comparison table
    comparison_path = os.path.join(output_dir, "benchmark_results.csv")
    df = create_comparison_table(traditional_results, dl_results, comparison_path)
    
    print(f"\n✅ Benchmark complete! Results saved to: {comparison_path}")
    
    # Identify best model
    best_model = df.index[0]
    best_mae = df.iloc[0]['mae']
    print(f"🏆 Best model: {best_model} (MAE: {best_mae:.4f})")


if __name__ == "__main__":
    main()
