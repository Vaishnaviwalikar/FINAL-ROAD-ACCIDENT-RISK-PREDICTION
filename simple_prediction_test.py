#!/usr/bin/env python3
"""
Simple test to check if predictions are real or simulated.
"""
import os
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_predictions():
    """Test predictions to see if they're real or simulated."""
    
    print("TESTING MODEL PREDICTIONS")
    print("=" * 50)
    
    try:
        # Import the prediction function
        from app_hybrid import load_model_and_preprocessing, predict_risk, a_model, a_feature_columns
        
        print("Loading model...")
        load_model_and_preprocessing()
        
        if a_model is None:
            print("NO MODEL LOADED - Using fallback predictions")
            model_status = "FALLBACK"
        else:
            print(f"MODEL LOADED: {type(a_model).__name__}")
            model_status = "REAL_MODEL"
            
        print(f"Features available: {len(a_feature_columns) if a_feature_columns else 0}")
        
        # Test with identical inputs multiple times
        test_input = {
            "Latitude": 25.0000,
            "Longitude": 77.0000,
            "hour": 12,
            "Day_of_Week": 2,
            "Weather_Conditions": 1,
            "Speed_limit": 40
        }
        
        print("\nTesting with identical inputs 3 times:")
        print(f"Input: {test_input}")
        
        results = []
        for i in range(3):
            try:
                prediction = predict_risk(test_input)
                print(f"\nTest {i+1}:")
                print(f"  Risk Value: {prediction.get('risk_value', 'N/A')}")
                print(f"  Risk Level: {prediction.get('risk_level', 'N/A')}")
                print(f"  Used Model: {prediction.get('used_model', False)}")
                print(f"  Source: {prediction.get('prediction_source', 'unknown')}")
                
                results.append(prediction)
                
            except Exception as e:
                print(f"  ERROR: {e}")
        
        # Analysis
        print("\nANALYSIS:")
        print("-" * 30)
        
        if len(results) >= 2:
            # Check if results are identical (deterministic)
            risk_values = [r.get('risk_value') for r in results if r.get('risk_value') is not None]
            
            if len(set(risk_values)) == 1:
                print("DETERMINISTIC: Same inputs give same outputs")
                print("This suggests REAL MODEL predictions")
            else:
                print("NON-DETERMINISTIC: Same inputs give different outputs")
                print("This suggests SIMULATED/RANDOM predictions")
                
            print(f"Risk values: {risk_values}")
            
            # Check prediction sources
            sources = [r.get('prediction_source', 'unknown') for r in results]
            print(f"Prediction sources: {set(sources)}")
            
            # Check if using model
            using_model = [r.get('used_model', False) for r in results]
            if any(using_model):
                print("STATUS: Using trained model")
            else:
                print("STATUS: Using fallback/simulated predictions")
        
        return model_status, results
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR", []

def check_model_files():
    """Check what model files exist."""
    print("\nCHECKING MODEL FILES:")
    print("-" * 30)
    
    model_paths = [
        "outputs/quick_fixed/best.pt",
        "outputs/final_fixed/best.pt", 
        "outputs/expanded_dataset_fixed/best.pt",
        "outputs/uk_full_cpu/best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"FOUND: {path} ({size_mb:.2f} MB)")
            
            try:
                checkpoint = torch.load(path, map_location='cpu')
                print(f"  Keys: {list(checkpoint.keys())}")
                if 'features' in checkpoint:
                    print(f"  Features: {len(checkpoint['features'])}")
            except Exception as e:
                print(f"  Error loading: {e}")
        else:
            print(f"NOT FOUND: {path}")

def main():
    print("ROAD TRAFFIC PREDICTION VERIFICATION")
    print("=" * 50)
    
    # Check files
    check_model_files()
    
    # Test predictions
    model_status, results = test_predictions()
    
    # Final verdict
    print("\nFINAL VERDICT:")
    print("=" * 30)
    
    if model_status == "REAL_MODEL":
        print("REAL AI MODEL is being used")
        print("Predictions are from trained neural network")
    elif model_status == "FALLBACK":
        print("FALLBACK/SIMULATED predictions are being used")
        print("No trained model is loaded")
    else:
        print("ERROR or unknown state")
    
    print("\nRECOMMENDATION:")
    if model_status == "REAL_MODEL":
        print("Add 'AI Model' indicator to UI")
    else:
        print("Add 'Simulated Data' indicator to UI")
        print("Or fix model loading to use real AI")

if __name__ == "__main__":
    main()