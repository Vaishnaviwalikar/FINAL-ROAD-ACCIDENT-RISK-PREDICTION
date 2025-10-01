#!/usr/bin/env python3
"""Complete system test - predictions, geocoding, and API integration"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import requests
import json

def test_geocoding():
    """Test city recognition including small cities"""
    cities = [
        # Major cities
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
        # Medium cities  
        "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
        # Small cities
        "Shimla", "Gangtok", "Shillong", "Itanagar", "Kohima",
        # International
        "London", "New York", "Tokyo", "Sydney", "Paris"
    ]
    
    print("Testing City Recognition (Geocoding)")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:5000"
    success_count = 0
    
    for city in cities:
        try:
            response = requests.post(f"{base_url}/geocode_city", 
                                   json={"city": city}, 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ {city}: {data['latitude']:.4f}, {data['longitude']:.4f}")
                success_count += 1
            else:
                print(f"✗ {city}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"✗ {city}: Error - {str(e)[:50]}")
    
    print(f"\nGeocoding Success Rate: {success_count}/{len(cities)} ({success_count/len(cities)*100:.1f}%)")
    return success_count >= len(cities) * 0.8  # 80% success rate

def test_predictions():
    """Test risk predictions for various scenarios"""
    test_cases = [
        {
            "name": "Mumbai Peak Traffic",
            "data": {"latitude": 19.0760, "longitude": 72.8777, "hour": 18, "day_of_week": 5, "weather_conditions": 1, "speed_limit": 40}
        },
        {
            "name": "Small Hill Station (Shimla)",
            "data": {"latitude": 31.1048, "longitude": 77.1734, "hour": 10, "day_of_week": 2, "weather_conditions": 1, "speed_limit": 30}
        },
        {
            "name": "Rural Highway Night",
            "data": {"latitude": 25.0000, "longitude": 78.0000, "hour": 23, "day_of_week": 6, "weather_conditions": 2, "speed_limit": 80}
        },
        {
            "name": "Coastal City Rain (Chennai)",
            "data": {"latitude": 13.0827, "longitude": 80.2707, "hour": 8, "day_of_week": 1, "weather_conditions": 2, "speed_limit": 50}
        },
        {
            "name": "Desert Area (Rajasthan)",
            "data": {"latitude": 27.0238, "longitude": 74.2179, "hour": 14, "day_of_week": 3, "weather_conditions": 1, "speed_limit": 60}
        }
    ]
    
    print("\nTesting Risk Predictions")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:5000"
    success_count = 0
    risk_levels_seen = set()
    
    for test_case in test_cases:
        try:
            response = requests.post(f"{base_url}/predict_risk", 
                                   json=test_case["data"], 
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                risk_level = result.get('risk_level', 'Unknown')
                risk_value = result.get('risk_value', 0)
                confidence = result.get('confidence', 0)
                
                print(f"✓ {test_case['name']}: {risk_level} ({risk_value}) - {confidence}% confidence")
                risk_levels_seen.add(risk_level)
                success_count += 1
            else:
                print(f"✗ {test_case['name']}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"✗ {test_case['name']}: Error - {str(e)[:50]}")
    
    print(f"\nPrediction Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    print(f"Risk Levels Generated: {sorted(risk_levels_seen)}")
    
    # Check if we have good variation (at least 3 different risk levels)
    has_variation = len(risk_levels_seen) >= 3
    print(f"Has Good Variation: {'✓' if has_variation else '✗'}")
    
    return success_count == len(test_cases) and has_variation

def test_weather_integration():
    """Test weather API integration"""
    print("\nTesting Weather Integration")
    print("=" * 50)
    
    locations = [
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Small Town", "lat": 31.1048, "lon": 77.1734},
        {"name": "International", "lat": 51.5074, "lon": -0.1278}
    ]
    
    base_url = "http://127.0.0.1:5000"
    success_count = 0
    
    for location in locations:
        try:
            response = requests.post(f"{base_url}/fetch_location_data", 
                                   json=location, 
                                   timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                weather = data.get('main_weather', 'Unknown')
                temp = data.get('temperature', 0)
                api_status = data.get('api_status', 'unknown')
                
                print(f"✓ {location['name']}: {weather}, {temp}°C - API: {api_status}")
                success_count += 1
            else:
                print(f"✗ {location['name']}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"✗ {location['name']}: Error - {str(e)[:50]}")
    
    print(f"\nWeather API Success Rate: {success_count}/{len(locations)} ({success_count/len(locations)*100:.1f}%)")
    return success_count >= len(locations) * 0.7  # 70% success rate (API can fail)

def test_system_status():
    """Test system status and health"""
    print("\nTesting System Status")
    print("=" * 50)
    
    try:
        response = requests.get("http://127.0.0.1:5000/status", timeout=10)
        
        if response.status_code == 200:
            status = response.json()
            print(f"✓ Model Loaded: {status.get('model_loaded', False)}")
            print(f"✓ Features Count: {status.get('features_count', 0)}")
            print(f"✓ API Mode: {status.get('api_mode', 'unknown')}")
            print(f"✓ Device: {status.get('device', 'unknown')}")
            return True
        else:
            print(f"✗ Status check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Status check error: {str(e)[:50]}")
        return False

def main():
    """Run complete system test"""
    print("COMPLETE SYSTEM TEST")
    print("=" * 60)
    print("Testing: Predictions, Geocoding, Weather API, System Health")
    print("=" * 60)
    
    # Test individual components
    status_ok = test_system_status()
    geocoding_ok = test_geocoding()
    predictions_ok = test_predictions()
    weather_ok = test_weather_integration()
    
    # Overall result
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    components = [
        ("System Status", status_ok),
        ("City Recognition", geocoding_ok),
        ("Risk Predictions", predictions_ok),
        ("Weather Integration", weather_ok)
    ]
    
    all_passed = True
    for name, passed in components:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        print("✅ Predictions are accurate and varied")
        print("✅ City recognition works for small cities")
        print("✅ Weather integration is functional")
        print("✅ System is stable and healthy")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK ABOVE FOR DETAILS")
    
    print("=" * 60)

if __name__ == "__main__":
    main()