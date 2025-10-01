import requests

print('Testing prediction API directly with extreme locations:')
print('=' * 60)

# Test with very different locations to show risk variation
test_cases = [
    {'name': 'Arctic', 'lat': 78.0, 'weather': 7},  # Snow, very northern
    {'name': 'Tropical', 'lat': 5.0, 'weather': 3},  # Rain, very tropical
    {'name': 'Desert', 'lat': 25.0, 'weather': 1},   # Clear, temperate
    {'name': 'London', 'lat': 51.5, 'weather': 1},   # Clear, northern
    {'name': 'Mumbai', 'lat': 19.1, 'weather': 3},   # Rain, tropical
]

for case in test_cases:
    try:
        data = {
            'latitude': case['lat'],
            'longitude': 0.0,
            'weather_conditions': case['weather'],
            'hour': 12,
            'speed_limit': 30,
            'road_surface': 0
        }

        response = requests.post('http://localhost:5000/predict_risk', json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(case['name'] + ': ' + result['risk_level'] + ' (' + str(round(result['risk_value'], 1)) + ') - ' + result['prediction_source'])
        else:
            print(case['name'] + ': ERROR ' + str(response.status_code))

    except Exception as e:
        print(case['name'] + ': Exception - ' + str(e)[:40] + '...')

print('')
print('If you see different risk values above, the system is working correctly!')
print('The issue might be that your browser is caching results or using demo mode.')
