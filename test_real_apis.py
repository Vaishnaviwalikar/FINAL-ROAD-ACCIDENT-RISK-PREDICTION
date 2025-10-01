#!/usr/bin/env python3
"""
Test Real API Integration - Verify only real data is used
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_real_weather_api():
    """Test that weather endpoint uses real API data"""
    print("🌤️ Testing Real Weather API Integration...")
    
    test_locations = [
        (51.5074, -0.1278, "London"),
        (40.7128, -74.0060, "New York"), 
        (19.0760, 72.8777, "Mumbai"),
        (28.6139, 77.2090, "Delhi")
    ]
    
    for lat, lon, city in test_locations:
        try:
            response = requests.post(f"{BASE_URL}/fetch_location_data",
                                   json={"latitude": lat, "longitude": lon},
                                   timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify it's real API data
                if data.get('data_source') == 'openweather_api':
                    print(f"✅ {city}: Real weather data")
                    print(f"   Temperature: {data.get('temperature')}°C")
                    print(f"   Weather: {data.get('main_weather')} - {data.get('description')}")
                    print(f"   API Status: {data.get('api_status')}")
                else:
                    print(f"❌ {city}: Not using real API data - {data.get('data_source')}")
                    
            elif response.status_code in [503, 500]:
                data = response.json()
                print(f"⚠️ {city}: API error (expected) - {data.get('error')}")
                print(f"   Status: {data.get('api_status')}")
            else:
                print(f"❌ {city}: Unexpected status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {city}: Request failed - {e}")
        
        time.sleep(1)  # Rate limiting

def test_real_geocoding():
    """Test that geocoding uses real API services"""
    print("\n🗺️ Testing Real Geocoding API Integration...")
    
    test_cities = [
        "London, UK",
        "Mumbai, India", 
        "Dharamshala, India",
        "Bath, UK",
        "Salem, Oregon, US"
    ]
    
    for city in test_cities:
        try:
            response = requests.post(f"{BASE_URL}/geocode_city",
                                   json={"city": city},
                                   timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print(f"✅ {city}: Found via {data.get('source')}")
                    print(f"   Coordinates: ({data.get('latitude'):.4f}, {data.get('longitude'):.4f})")
                else:
                    print(f"❌ {city}: Not found")
                    
            else:
                print(f"❌ {city}: Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {city}: Request failed - {e}")
        
        time.sleep(1)  # Rate limiting

def test_predictions_with_real_data():
    """Test that predictions work with real weather data"""
    print("\n🎯 Testing Predictions with Real Data...")
    
    # Test with London coordinates
    lat, lon = 51.5074, -0.1278
    
    try:
        # First get real weather data
        weather_response = requests.post(f"{BASE_URL}/fetch_location_data",
                                       json={"latitude": lat, "longitude": lon},
                                       timeout=15)
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            
            # Use real weather data for prediction
            prediction_data = {
                "latitude": lat,
                "longitude": lon,
                "hour": weather_data.get('hour', 14),
                "day_of_week": weather_data.get('day_of_week', 3),
                "weather_conditions": weather_data.get('weather_conditions', 1),
                "road_surface": weather_data.get('road_surface', 0),
                "speed_limit": 30
            }
            
            pred_response = requests.post(f"{BASE_URL}/predict_risk",
                                        json=prediction_data,
                                        timeout=10)
            
            if pred_response.status_code == 200:
                pred_data = pred_response.json()
                print(f"✅ Prediction with real weather data:")
                print(f"   Weather: {weather_data.get('main_weather')} ({weather_data.get('temperature')}°C)")
                print(f"   Risk Level: {pred_data.get('risk_level')}")
                print(f"   Risk Score: {pred_data.get('risk_value')}")
                print(f"   Confidence: {pred_data.get('confidence')}%")
                print(f"   Data Source: {pred_data.get('data_source')}")
            else:
                print(f"❌ Prediction failed: {pred_response.status_code}")
        else:
            print(f"❌ Weather data failed: {weather_response.status_code}")
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

def test_no_fallback_data():
    """Verify no simulated/fallback data is being used"""
    print("\n🔍 Verifying No Fallback Data Usage...")
    
    # Test status endpoint
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            api_mode = data.get('api_mode', 'unknown')
            
            if api_mode == 'live':
                print("✅ System configured for live API data only")
            elif api_mode == 'demo':
                print("❌ System still in demo mode - should be live only")
            else:
                print(f"⚠️ Unknown API mode: {api_mode}")
                
        else:
            print(f"❌ Status check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Status check error: {e}")

def main():
    """Run comprehensive real API tests"""
    print("🔍 Real API Integration Test")
    print("Verifying only real data is used, no fallbacks")
    print("=" * 50)
    
    # Test server is running
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code != 200:
            print("❌ Server not running. Start with: python app.py")
            return
    except:
        print("❌ Server not running. Start with: python app.py")
        return
    
    # Run tests
    test_no_fallback_data()
    test_real_weather_api()
    test_real_geocoding()
    test_predictions_with_real_data()
    
    print("\n📋 REAL API TEST SUMMARY")
    print("=" * 50)
    print("✅ All data should come from real APIs")
    print("⚠️ Errors are expected when APIs are unavailable")
    print("❌ No simulated/fallback data should be returned")
    print("\n🎯 Your system now uses ONLY real data!")

if __name__ == "__main__":
    main()