#!/usr/bin/env python3
"""
Verification script to check that all components are working correctly
"""
import sys
import os
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all modules import correctly"""
    print("🔍 Testing imports...")
    try:
        import torch
        print("✅ PyTorch imported")
        
        from src.model import CNNBiLSTMAttn, SimplifiedRiskModel
        print("✅ Model classes imported")
        
        from src.preprocess import main as preprocess_main
        print("✅ Preprocessing module imported")
        
        from src.train import main as train_main
        print("✅ Training module imported")
        
        from src.explain_deepshap import main as explain_main
        print("✅ DeepSHAP module imported")
        
        from config_api import get_openweather_api_key, is_demo_mode
        print("✅ API config imported")
        
        import app
        print("✅ Flask app imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_model_instantiation():
    """Test that models can be instantiated"""
    print("\n🔍 Testing model instantiation...")
    try:
        import torch
        from src.model import CNNBiLSTMAttn, SimplifiedRiskModel
        
        # Test CNN-BiLSTM-Attention
        model1 = CNNBiLSTMAttn(
            in_channels=10,
            cnn_channels=(32, 64),
            kernel_sizes=(3, 3),
            pool_size=2,
            fc_dim=64,
            attn_spatial_dim=32,
            attn_temporal_dim=32,
            lstm_hidden=64,
            lstm_layers=1,
            dropout=0.3
        )
        print("✅ CNNBiLSTMAttn instantiated")
        
        # Test forward pass
        x = torch.randn(2, 5, 10)  # batch=2, seq_len=5, features=10
        with torch.no_grad():
            out = model1(x)
        print(f"✅ CNNBiLSTMAttn forward pass: {out.shape}")
        
        # Test SimplifiedRiskModel
        model2 = SimplifiedRiskModel(in_channels=10, hidden_dim=32, dropout=0.5)
        with torch.no_grad():
            out2 = model2(x)
        print(f"✅ SimplifiedRiskModel forward pass: {out2.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model instantiation error: {e}")
        traceback.print_exc()
        return False

def test_api_config():
    """Test API configuration"""
    print("\n🔍 Testing API configuration...")
    try:
        from config_api import get_openweather_api_key, is_demo_mode
        
        api_key = get_openweather_api_key()
        demo_mode = is_demo_mode()
        
        print(f"✅ API key retrieved: {'demo_key' if demo_mode else 'real key'}")
        print(f"✅ Demo mode: {demo_mode}")
        
        return True
    except Exception as e:
        print(f"❌ API config error: {e}")
        traceback.print_exc()
        return False

def test_flask_routes():
    """Test Flask app routes"""
    print("\n🔍 Testing Flask routes...")
    try:
        import app
        
        client = app.app.test_client()
        
        # Test status endpoint
        resp = client.get('/status')
        print(f"✅ /status: {resp.status_code}")
        
        # Test predict_risk endpoint
        payload = {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "hour": 12,
            "day_of_week": 3,
            "month": 6,
            "weather_conditions": 1,
            "road_surface": 0,
            "speed_limit": 30
        }
        resp = client.post('/predict_risk', json=payload)
        print(f"✅ /predict_risk: {resp.status_code}")
        
        # Test fetch_location_data
        resp = client.post('/fetch_location_data', json={"latitude": 51.5, "longitude": -0.1})
        print(f"✅ /fetch_location_data: {resp.status_code}")
        
        # Test geocode_city
        resp = client.post('/geocode_city', json={"city": "London"})
        print(f"✅ /geocode_city: {resp.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Flask routes error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("🚀 Starting implementation verification...\n")
    
    tests = [
        test_imports,
        test_model_instantiation,
        test_api_config,
        test_flask_routes
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All verification tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
