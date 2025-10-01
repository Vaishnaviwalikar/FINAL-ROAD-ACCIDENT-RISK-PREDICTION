import requests
import json
from datetime import datetime

# Test prediction API with different cities and their weather data
def test_prediction_api():
    # Simulate different cities with their coordinates and expected weather data
    test_cases = [
        {
            'city': 'London',
            'lat': 51.5074,
            'lon': -0.1278,
            'expected_weather': 'Clear',
            'expected_temp': 10
        },
        {
            'city': 'Mumbai',
            'lat': 19.0760,
            'lon': 72.8777,
            'expected_weather': 'Clouds',
            'expected_temp': 28
        },
        {
            'city': 'New York',
            'lat': 40.7128,
            'lon': -74.0060,
            'expected_weather': 'Clear',
            'expected_temp': 15
        }
    ]

    print("Testing prediction API with different cities:")
    print("=" * 60)

    for test_case in test_cases:
        print(f"\n🗺️  Testing {test_case['city']}...")

        # Simulate the data that would be sent to /predict_risk
        # This should include location and current weather conditions
        prediction_data = {
            'latitude': test_case['lat'],
            'longitude': test_case['lon'],
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().isoweekday(),
            'month': datetime.now().month,
            'weather_conditions': 1,  # Will be updated by weather API
            'road_surface': 0,
            'speed_limit': 30,
            'road_type': 3,
            'junction_detail': 0
        }

        print(f"   Input data: {prediction_data}")

        try:
            response = requests.post(
                'http://localhost:5000/predict_risk',
                json=prediction_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: Risk Level = {result.get('risk_level', 'N/A')}")
                print(f"   📊 Risk Value = {result.get('risk_value', 'N/A')}")
                print(f"   🎯 Confidence = {result.get('confidence', 'N/A')}%")
                print(f"   🔧 Used Model = {result.get('used_model', False)}")
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")

        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")

    print("\n" + "=" * 60)
    print("Test completed! Check if different cities show different risk levels.")

if __name__ == "__main__":
    test_prediction_api()
