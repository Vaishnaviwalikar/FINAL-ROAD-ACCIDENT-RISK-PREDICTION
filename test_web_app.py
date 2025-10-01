#!/usr/bin/env python3
"""
Test script to verify the web application is working correctly.
"""
import requests
import json
import time

def test_endpoints():
    """Test all main endpoints of the web application."""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing RoadSafe AI Web Application")
    print("=" * 50)
    
    # Test 1: Status endpoint
    print("\n1. Testing /status endpoint...")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   📊 Model loaded: {status.get('model_loaded', False)}")
            print(f"   🔧 Features count: {status.get('features_count', 0)}")
            print(f"   🌐 API mode: {status.get('api_mode', 'unknown')}")
        else:
            print(f"   ❌ Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Status endpoint error: {e}")
        return False
    
    # Test 2: Prediction endpoint
    print("\n2. Testing /predict_risk endpoint...")
    try:
        test_payload = {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "hour": 14,
            "day_of_week": 3,
            "weather_conditions": 1,
            "road_surface": 0,
            "speed_limit": 30
        }
        
        response = requests.post(
            f"{base_url}/predict_risk",
            headers={"Content-Type": "application/json"},
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction: {response.status_code}")
            print(f"   🎯 Risk level: {result.get('risk_level', 'Unknown')}")
            print(f"   📈 Risk value: {result.get('risk_value', 0)}")
            print(f"   🔍 Confidence: {result.get('confidence', 0)}%")
            print(f"   🧠 Used AI model: {result.get('used_model', False)}")
            print(f"   📡 Source: {result.get('prediction_source', 'unknown')}")
            
            if result.get('used_model'):
                print("   🎉 SUCCESS: Real AI model is being used!")
            else:
                print("   ⚠️  WARNING: Fallback mode active")
                
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction endpoint error: {e}")
        return False
    
    # Test 3: Geocoding endpoint
    print("\n3. Testing /geocode_city endpoint...")
    try:
        response = requests.post(
            f"{base_url}/geocode_city",
            headers={"Content-Type": "application/json"},
            json={"city": "London"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Geocoding: {response.status_code}")
            print(f"   📍 City: {result.get('city', 'Unknown')}")
            print(f"   🌍 Coordinates: {result.get('latitude', 0)}, {result.get('longitude', 0)}")
        else:
            print(f"   ❌ Geocoding failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Geocoding endpoint error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Web application test completed!")
    return True

if __name__ == "__main__":
    print("Starting web application test...")
    print("Make sure the Flask app is running on localhost:5000")
    print("You can start it with: python app_hybrid.py")
    print("\nWaiting 3 seconds before testing...")
    time.sleep(3)
    
    success = test_endpoints()
    if success:
        print("\n✅ All tests passed! The web application is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the Flask application logs.")