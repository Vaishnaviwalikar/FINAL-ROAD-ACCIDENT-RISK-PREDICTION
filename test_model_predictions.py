#!/usr/bin/env python3
"""
Test script to verify model predictions and identify if they're real or simulated.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_predictions():
    """Test the model with various inputs to check if predictions are real."""
    
    print("🔍 TESTING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Try to load the model using the same method as the app
    try:
        from app_hybrid import load_model_and_preprocessing, predict_risk, a_model, a_feature_columns
        
        print("📥 Loading model...")
        load_model_and_preprocessing()
        
        if a_model is None:
            print("❌ No model loaded - predictions will be fallback/simulated")
            model_type = "FALLBACK/SIMULATED"
        else:
            print(f"✅ Model loaded: {type(a_model).__name__}")
            model_type = "REAL AI MODEL"
            
        print(f"📊 Features available: {len(a_feature_columns) if a_feature_columns else 0}")
        
        # Test with different input scenarios
        test_cases = [
            {
                "name": "London Rush Hour",
                "data": {
                    "Latitude": 51.5074,
                    "Longitude": -0.1278,
                    "hour": 8,
                    "Day_of_Week": 1,  # Monday
                    "Weather_Conditions": 1,  # Clear
                    "Speed_limit": 30
                }
            },
            {
                "name": "Mumbai Night Rain",
                "data": {
                    "Latitude": 19.0760,
                    "Longitude": 72.8777,
                    "hour": 23,
                    "Day_of_Week": 5,  # Friday
                    "Weather_Conditions": 3,  # Rain
                    "Speed_limit": 50
                }
            },
            {
                "name": "New York Highway",
                "data": {
                    "Latitude": 40.7128,
                    "Longitude": -74.0060,
                    "hour": 14,
                    "Day_of_Week": 3,  # Wednesday
                    "Weather_Conditions": 1,  # Clear
                    "Speed_limit": 70
                }
            },
            {
                "name": "Same Location Test 1",
                "data": {
                    "Latitude": 25.0000,
                    "Longitude": 77.0000,
                    "hour": 12,
                    "Day_of_Week": 2,
                    "Weather_Conditions": 1,
                    "Speed_limit": 40
                }
            },
            {
                "name": "Same Location Test 2",
                "data": {
                    "Latitude": 25.0000,
                    "Longitude": 77.0000,
                    "hour": 12,
                    "Day_of_Week": 2,
                    "Weather_Conditions": 1,
                    "Speed_limit": 40
                }
            }
        ]
        
        print(f"\n🧪 TESTING WITH {model_type}")
        print("-" * 60)
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}:")
            print(f"   Input: {test_case['data']}")
            
            try:
                prediction = predict_risk(test_case['data'])
                print(f"   Result: {prediction}")
                
                results.append({
                    "test": test_case['name'],
                    "risk_value": prediction.get('risk_value', 'N/A'),
                    "risk_level": prediction.get('risk_level', 'N/A'),
                    "confidence": prediction.get('confidence', 'N/A'),
                    "used_model": prediction.get('used_model', False),
                    "source": prediction.get('prediction_source', 'unknown')
                })
                
            except Exception as e:
                print(f"   ERROR: {e}")
                results.append({
                    "test": test_case['name'],
                    "error": str(e)
                })
        
        # Analysis
        print(f"\n📈 ANALYSIS")
        print("-" * 60)
        
        # Check if same inputs give same outputs (deterministic)
        same_location_results = [r for r in results if 'Same Location Test' in r.get('test', '')]
        if len(same_location_results) >= 2:
            result1 = same_location_results[0]
            result2 = same_location_results[1]
            
            if result1.get('risk_value') == result2.get('risk_value'):
                print("✅ DETERMINISTIC: Same inputs produce same outputs (likely real model)")
            else:
                print("⚠️  NON-DETERMINISTIC: Same inputs produce different outputs (likely simulated)")
        
        # Check prediction sources
        sources = [r.get('source', 'unknown') for r in results if 'error' not in r]
        unique_sources = set(sources)
        print(f"\n📊 Prediction Sources Found: {unique_sources}")
        
        # Check if using real model
        using_model = [r.get('used_model', False) for r in results if 'error' not in r]
        if any(using_model):
            print("✅ REAL MODEL: At least some predictions use trained model")
        else:
            print("❌ NO REAL MODEL: All predictions are fallback/simulated")
        
        # Check risk value patterns
        risk_values = [r.get('risk_value') for r in results if 'error' not in r and r.get('risk_value') != 'N/A']
        if risk_values:
            print(f"\n🎯 Risk Values: {risk_values}")
            print(f"   Range: {min(risk_values):.3f} - {max(risk_values):.3f}")
            print(f"   Variance: {np.var(risk_values):.6f}")
            
            if np.var(risk_values) < 0.001:
                print("⚠️  SUSPICIOUS: Very low variance in predictions (likely fake)")
            else:
                print("✅ GOOD: Reasonable variance in predictions")
        
        return results, model_type
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return [], "ERROR"

def check_model_files():
    """Check what model files exist and their properties."""
    print(f"\n📁 CHECKING MODEL FILES")
    print("-" * 60)
    
    model_dirs = [
        "outputs/quick_fixed",
        "outputs/final_fixed", 
        "outputs/expanded_dataset_fixed",
        "outputs/uk_full_cpu",
        "outputs/uk_top15_e50"
    ]
    
    for model_dir in model_dirs:
        model_path = os.path.join(model_dir, "best.pt")
        if os.path.exists(model_path):
            try:
                # Load checkpoint to inspect
                checkpoint = torch.load(model_path, map_location='cpu')
                
                print(f"\n📦 {model_dir}/best.pt:")
                print(f"   Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
                print(f"   Keys: {list(checkpoint.keys())}")
                
                if 'features' in checkpoint:
                    print(f"   Features: {len(checkpoint['features'])}")
                    print(f"   First 5 features: {checkpoint['features'][:5]}")
                
                if 'cfg' in checkpoint:
                    cfg = checkpoint['cfg']
                    print(f"   Config keys: {list(cfg.keys())}")
                    
                # Try to check if model state exists and is valid
                if 'model' in checkpoint:
                    model_state = checkpoint['model']
                    if isinstance(model_state, dict):
                        print(f"   Model parameters: {len(model_state)} layers")
                        # Check a few parameter shapes
                        param_info = []
                        for name, param in list(model_state.items())[:3]:
                            param_info.append(f"{name}: {param.shape}")
                        print(f"   Sample params: {param_info}")
                    else:
                        print(f"   Model state type: {type(model_state)}")
                        
            except Exception as e:
                print(f"   ERROR loading {model_path}: {e}")
        else:
            print(f"❌ {model_path} - Not found")

def main():
    """Main test function."""
    print("🚗 ROAD TRAFFIC PREDICTION MODEL VERIFICATION")
    print("=" * 80)
    
    # Check model files first
    check_model_files()
    
    # Test predictions
    results, model_type = test_model_predictions()
    
    # Final verdict
    print(f"\n🏁 FINAL VERDICT")
    print("=" * 60)
    
    if model_type == "REAL AI MODEL":
        print("✅ REAL MODEL DETECTED")
        print("   - Trained PyTorch model is loaded and being used")
        print("   - Predictions should be based on learned patterns")
        print("   - Model was trained on actual traffic accident data")
    elif model_type == "FALLBACK/SIMULATED":
        print("⚠️  FALLBACK MODE DETECTED")
        print("   - No trained model loaded")
        print("   - Using rule-based/simulated predictions")
        print("   - Predictions are based on heuristics, not ML")
    else:
        print("❌ ERROR OR UNKNOWN STATE")
        print("   - Could not determine prediction method")
        print("   - Check logs for errors")
    
    print(f"\n💡 RECOMMENDATION:")
    if model_type == "REAL AI MODEL":
        print("   The system is using real AI predictions. Add a data source")
        print("   indicator to the UI to show users this is AI-powered.")
    else:
        print("   The system is using simulated predictions. Either:")
        print("   1. Fix model loading issues, or")
        print("   2. Clearly label predictions as 'Simulated' or 'Demo Mode'")

if __name__ == "__main__":
    main()