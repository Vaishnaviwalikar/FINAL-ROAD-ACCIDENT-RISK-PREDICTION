#!/usr/bin/env python3
"""
Quick test to verify prediction improvements
"""
import requests
import json

def quick_test():
    base_url = "http://127.0.0.1:5000"
    
    print("🔧 Quick Prediction Test")
    print("=" * 40)
    
    # Test a few different locations
    locations = [
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    ]
    
    for loc in locations:
        try:
            payload = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "hour": 14,
                "weather_conditions": 1,
                "road_surface": 0,
                "speed_limit": 30
            }
            
            response = requests.post(f"{base_url}/predict_risk", json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                print(f"{loc['name']:8} | Risk: {result['risk_level']:6} ({result['risk_value']}) | "
                      f"Conf: {result['confidence']:4.1f}% | "
                      f"{'Model' if result.get('used_model') else 'Fallback'}")
            else:
                print(f"{loc['name']:8} | ERROR: {response.status_code}")
                
        except Exception as e:
            print(f"{loc['name']:8} | EXCEPTION: {str(e)[:30]}...")
    
    print("\n✅ Test complete! Check if predictions vary by location.")

if __name__ == "__main__":
    quick_test()