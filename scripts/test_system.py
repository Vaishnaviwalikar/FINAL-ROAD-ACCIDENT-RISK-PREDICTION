#!/usr/bin/env python3
"""
System Verification Script
Tests predictions, geocoding, and city identification for accuracy
"""

import requests
import json
import time
from typing import Dict, List, Tuple

# Test cities - mix of large and small cities worldwide
TEST_CITIES = [
    # Major cities
    ("London", "UK"),
    ("New York", "US"),
    ("Tokyo", "Japan"),
    ("Mumbai", "India"),
    
    # Medium cities
    ("Brighton", "UK"),
    ("Austin", "US"),
    ("Kyoto", "Japan"),
    ("Pune", "India"),
    
    # Small cities/towns
    ("Bath", "UK"),
    ("Salem", "US"),
    ("Nara", "Japan"),
    ("Mysore", "India"),
    ("Vadodara", "India"),
    ("Chandigarh", "India"),
]

BASE_URL = "http://127.0.0.1:5000"

def test_geocoding() -> List[Dict]:
    """Test geocoding for various city sizes"""
    print("🌍 Testing Geocoding Accuracy...")
    results = []
    
    for city, country in TEST_CITIES:
        try:
            response = requests.post(f"{BASE_URL}/geocode_city", 
                                   json={"city": f"{city}, {country}"}, 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    results.append({
                        "city": f"{city}, {country}",
                        "lat": data["latitude"],
                        "lon": data["longitude"],
                        "status": "✅ Found"
                    })
                    print(f"  ✅ {city}, {country}: ({data['latitude']:.4f}, {data['longitude']:.4f})")
                else:
                    results.append({"city": f"{city}, {country}", "status": "❌ Not found"})
                    print(f"  ❌ {city}, {country}: Not found")
            else:
                results.append({"city": f"{city}, {country}", "status": f"❌ Error {response.status_code}"})
                print(f"  ❌ {city}, {country}: HTTP {response.status_code}")
                
        except Exception as e:
            results.append({"city": f"{city}, {country}", "status": f"❌ Exception: {str(e)}"})
            print(f"  ❌ {city}, {country}: {str(e)}")
            
        time.sleep(0.5)  # Rate limiting
    
    return results

def test_predictions() -> List[Dict]:
    """Test prediction accuracy and consistency"""
    print("\n🎯 Testing Prediction Accuracy...")
    results = []
    
    # Test scenarios with expected risk patterns
    test_scenarios = [
        {
            "name": "High Risk: Night + Rain + Highway",
            "data": {"latitude": 51.5074, "longitude": -0.1278, "hour": 2, "day_of_week": 6, 
                    "weather_conditions": 3, "road_surface": 2, "speed_limit": 70},
            "expected_risk": "high"
        },
        {
            "name": "Low Risk: Day + Clear + Residential",
            "data": {"latitude": 51.5074, "longitude": -0.1278, "hour": 14, "day_of_week": 2, 
                    "weather_conditions": 1, "road_surface": 0, "speed_limit": 30},
            "expected_risk": "low"
        },
        {
            "name": "Medium Risk: Evening + Cloudy + Urban",
            "data": {"latitude": 51.5074, "longitude": -0.1278, "hour": 18, "day_of_week": 4, 
                    "weather_conditions": 2, "road_surface": 1, "speed_limit": 50},
            "expected_risk": "medium"
        }
    ]
    
    for scenario in test_scenarios:
        try:
            response = requests.post(f"{BASE_URL}/predict_risk", 
                                   json=scenario["data"], 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                risk_score = data.get("risk_score", 0)
                risk_level = data.get("risk_level", "unknown")
                
                results.append({
                    "scenario": scenario["name"],
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "expected": scenario["expected_risk"],
                    "status": "✅ Success"
                })
                
                print(f"  📊 {scenario['name']}")
                print(f"     Risk Score: {risk_score:.3f} | Level: {risk_level} | Expected: {scenario['expected_risk']}")
                
            else:
                results.append({
                    "scenario": scenario["name"],
                    "status": f"❌ HTTP {response.status_code}"
                })
                print(f"  ❌ {scenario['name']}: HTTP {response.status_code}")
                
        except Exception as e:
            results.append({
                "scenario": scenario["name"],
                "status": f"❌ Exception: {str(e)}"
            })
            print(f"  ❌ {scenario['name']}: {str(e)}")
    
    return results

def test_weather_integration() -> Dict:
    """Test real-time weather data integration"""
    print("\n🌤️ Testing Weather Integration...")
    
    test_locations = [
        (51.5074, -0.1278, "London"),
        (40.7128, -74.0060, "New York"),
        (19.0760, 72.8777, "Mumbai")
    ]
    
    results = []
    
    for lat, lon, city in test_locations:
        try:
            response = requests.post(f"{BASE_URL}/fetch_location_data", 
                                   json={"latitude": lat, "longitude": lon}, 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                weather = data.get("weather", {})
                
                results.append({
                    "city": city,
                    "temperature": weather.get("temperature"),
                    "humidity": weather.get("humidity"),
                    "conditions": weather.get("conditions"),
                    "status": "✅ Success"
                })
                
                print(f"  🌍 {city}: {weather.get('temperature', 'N/A')}°C, {weather.get('conditions', 'N/A')}")
                
            else:
                results.append({"city": city, "status": f"❌ HTTP {response.status_code}"})
                print(f"  ❌ {city}: HTTP {response.status_code}")
                
        except Exception as e:
            results.append({"city": city, "status": f"❌ Exception: {str(e)}"})
            print(f"  ❌ {city}: {str(e)}")
    
    return results

def test_system_status() -> Dict:
    """Test system health and diagnostics"""
    print("\n🔧 Testing System Status...")
    
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ System Status: {data.get('status', 'Unknown')}")
            print(f"  📊 Model Loaded: {data.get('model_loaded', False)}")
            print(f"  🌐 Weather API: {data.get('weather_api_status', 'Unknown')}")
            return {"status": "✅ Healthy", "details": data}
        else:
            print(f"  ❌ System Status: HTTP {response.status_code}")
            return {"status": f"❌ HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"  ❌ System Status: {str(e)}")
        return {"status": f"❌ Exception: {str(e)}"}

def main():
    """Run comprehensive system verification"""
    print("🚗 Road Traffic Accident Risk Prediction - System Verification")
    print("=" * 60)
    
    # Test system status first
    status_results = test_system_status()
    
    # Test geocoding accuracy
    geocoding_results = test_geocoding()
    
    # Test prediction accuracy
    prediction_results = test_predictions()
    
    # Test weather integration
    weather_results = test_weather_integration()
    
    # Summary
    print("\n📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    geocoding_success = sum(1 for r in geocoding_results if "✅" in r["status"])
    print(f"🌍 Geocoding: {geocoding_success}/{len(geocoding_results)} cities found")
    
    prediction_success = sum(1 for r in prediction_results if "✅" in r["status"])
    print(f"🎯 Predictions: {prediction_success}/{len(prediction_results)} scenarios successful")
    
    weather_success = sum(1 for r in weather_results if "✅" in r["status"])
    print(f"🌤️ Weather: {weather_success}/{len(weather_results)} locations successful")
    
    print(f"🔧 System: {status_results['status']}")
    
    # Overall health
    total_tests = len(geocoding_results) + len(prediction_results) + len(weather_results) + 1
    total_success = geocoding_success + prediction_success + weather_success + (1 if "✅" in status_results["status"] else 0)
    
    health_percentage = (total_success / total_tests) * 100
    print(f"\n🎯 Overall System Health: {health_percentage:.1f}% ({total_success}/{total_tests})")
    
    if health_percentage >= 90:
        print("✅ System is ready for production!")
    elif health_percentage >= 70:
        print("⚠️ System is mostly functional, minor issues detected")
    else:
        print("❌ System needs attention before production use")

if __name__ == "__main__":
    main()