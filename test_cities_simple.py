import requests
import json

# Test prediction with different cities
test_cities = [
    {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'weather': 1},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'weather': 3},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'weather': 1}
]

print('Testing prediction API with different cities:')
print('=' * 60)

for city in test_cities:
    try:
        data = {
            'latitude': city['lat'],
            'longitude': city['lon'],
            'weather_conditions': city['weather'],
            'hour': 14,  # 2 PM
            'speed_limit': 30
        }

        response = requests.post('http://localhost:5000/predict_risk', json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            risk_level = result.get('risk_level', 'N/A')
            risk_value = result.get('risk_value', 0)
            source = result.get('prediction_source', 'unknown')
            print(city['name'] + ': ' + risk_level + ' (' + str(risk_value) + ') - ' + source)
        else:
            print(city['name'] + ': ERROR ' + str(response.status_code))

    except Exception as e:
        print(city['name'] + ': Exception - ' + str(e)[:30] + '...')

print('')
print('Test completed!')
