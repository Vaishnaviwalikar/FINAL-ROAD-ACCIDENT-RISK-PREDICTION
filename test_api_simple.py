import requests
import json

print("Testing RoadSafe AI API endpoints...")
print("=" * 50)

# Test the status endpoint
try:
    response = requests.get('http://localhost:5000/status', timeout=5)
    print(f'Status endpoint: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'  Model loaded: {data.get("model_loaded")}')
        print(f'  ML available: {data.get("ml_available")}')
        print(f'  Features count: {data.get("features_count")}')
    print()
except Exception as e:
    print(f'Status check failed: {e}')
    print()

# Test prediction with sample data
try:
    test_data = {
        'latitude': 51.5074,
        'longitude': -0.1278,
        'weather_conditions': 1,
        'road_surface': 0,
        'speed_limit': 30
    }

    response = requests.post('http://localhost:5000/predict_risk', json=test_data, timeout=5)
    print(f'Prediction endpoint: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'  Risk Level: {result.get("risk_level")}')
        print(f'  Risk Value: {result.get("risk_value")}')
        print(f'  Used Model: {result.get("used_model")}')
        print(f'  Prediction Source: {result.get("prediction_source")}')
    else:
        print(f'  Error response: {response.text[:200]}...')
        print()
except Exception as e:
    print(f'Prediction test failed: {e}')

print("Test completed!")
