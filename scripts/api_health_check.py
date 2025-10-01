#!/usr/bin/env python3
"""
API Health Check - Verify all APIs are working with real data
"""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def test_openweather_api():
    """Test OpenWeatherMap API with real calls"""
    print("🌤️ Testing OpenWeatherMap API...")
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        print("❌ OPENWEATHER_API_KEY not found in environment")
        return False
    
    # Test with London coordinates
    lat, lon = 51.5074, -0.1278
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ OpenWeatherMap API working")
            print(f"   Location: {data.get('name', 'Unknown')}")
            print(f"   Weather: {data['weather'][0]['main']} - {data['weather'][0]['description']}")
            print(f"   Temperature: {data['main']['temp']}°C")
            return True
        else:
            print(f"❌ OpenWeatherMap API error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ OpenWeatherMap API error: {e}")
        return False

def test_tomtom_api():
    """Test TomTom API with real calls"""
    print("\n🗺️ Testing TomTom API...")
    
    api_key = os.getenv('TOMTOM_API_KEY')
    if not api_key:
        print("❌ TOMTOM_API_KEY not found in environment")
        return False
    
    # Test geocoding
    city = "London"
    url = f"https://api.tomtom.com/search/2/geocode/{city}.json?key={api_key}&limit=1"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                pos = result['position']
                print(f"✅ TomTom API working")
                print(f"   Found: {result['address'].get('freeformAddress', city)}")
                print(f"   Coordinates: ({pos['lat']}, {pos['lon']})")
                return True
            else:
                print("❌ TomTom API: No results found")
                return False
        else:
            print(f"❌ TomTom API error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ TomTom API error: {e}")
        return False

def test_nominatim_api():
    """Test Nominatim (OpenStreetMap) API"""
    print("\n🌍 Testing Nominatim API...")
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': 'London, UK',
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }
    headers = {'User-Agent': 'RoadRiskPredictor/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                print(f"✅ Nominatim API working")
                print(f"   Found: {result.get('display_name', 'Unknown').split(',')[0]}")
                print(f"   Coordinates: ({result['lat']}, {result['lon']})")
                return True
            else:
                print("❌ Nominatim API: No results found")
                return False
        else:
            print(f"❌ Nominatim API error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Nominatim API error: {e}")
        return False

def test_api_rate_limits():
    """Test API rate limits and response times"""
    print("\n⏱️ Testing API Performance...")
    
    apis = [
        ("OpenWeatherMap", test_openweather_api),
        ("TomTom", test_tomtom_api),
        ("Nominatim", test_nominatim_api)
    ]
    
    for name, test_func in apis:
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        
        if success:
            print(f"   {name} response time: {end_time - start_time:.2f}s")
        
        # Rate limiting delay
        time.sleep(1)

def main():
    """Run comprehensive API health check"""
    print("🔍 API Health Check - Real Data Verification")
    print("=" * 50)
    
    # Check environment variables
    print("📋 Environment Variables:")
    openweather_key = os.getenv('OPENWEATHER_API_KEY')
    tomtom_key = os.getenv('TOMTOM_API_KEY')
    
    print(f"   OPENWEATHER_API_KEY: {'✅ Set' if openweather_key else '❌ Missing'}")
    print(f"   TOMTOM_API_KEY: {'✅ Set' if tomtom_key else '❌ Missing'}")
    
    if not openweather_key:
        print("\n❌ Critical: OPENWEATHER_API_KEY is required for weather data")
        return
    
    # Test each API
    results = []
    
    print(f"\n🧪 Testing APIs with Real Calls...")
    print("-" * 30)
    
    results.append(("OpenWeatherMap", test_openweather_api()))
    time.sleep(1)  # Rate limiting
    
    results.append(("TomTom", test_tomtom_api()))
    time.sleep(1)  # Rate limiting
    
    results.append(("Nominatim", test_nominatim_api()))
    
    # Summary
    print(f"\n📊 API Health Summary")
    print("=" * 30)
    
    working_apis = sum(1 for _, status in results if status)
    total_apis = len(results)
    
    for name, status in results:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}: {'Working' if status else 'Failed'}")
    
    print(f"\n🎯 Overall: {working_apis}/{total_apis} APIs working")
    
    if working_apis == total_apis:
        print("🎉 All APIs are working with real data!")
        print("✅ Your application will use live data from all sources")
    elif working_apis >= 2:
        print("⚠️ Most APIs working - fallbacks available")
        print("✅ Your application will work with some fallbacks")
    else:
        print("❌ Critical: Multiple API failures detected")
        print("🔧 Check your API keys and internet connection")
    
    print(f"\n📖 Next Steps:")
    if working_apis == total_apis:
        print("1. Start your app: python app.py")
        print("2. All data will be real-time from APIs")
        print("3. Fallbacks only used if APIs become unavailable")
    else:
        print("1. Fix API key issues above")
        print("2. Check internet connection")
        print("3. Re-run this test: python scripts/api_health_check.py")

if __name__ == "__main__":
    main()