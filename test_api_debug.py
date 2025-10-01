import requests

# Test the weather cache and API directly
print('Testing weather API cache behavior:')
print('=' * 50)

# Test London
response1 = requests.post('http://localhost:5000/fetch_location_data',
                         json={'latitude': 51.5074, 'longitude': -0.1278}, timeout=5)
if response1.status_code == 200:
    data1 = response1.json()
    print('London weather:', data1.get('weather_conditions'), '- Source:', data1.get('data_source'))

# Small delay to test caching
import time
time.sleep(1)

# Test Mumbai
response2 = requests.post('http://localhost:5000/fetch_location_data',
                         json={'latitude': 19.0760, 'longitude': 72.8777}, timeout=5)
if response2.status_code == 200:
    data2 = response2.json()
    print('Mumbai weather:', data2.get('weather_conditions'), '- Source:', data2.get('data_source'))

# Test same location again (should use cache)
response3 = requests.post('http://localhost:5000/fetch_location_data',
                         json={'latitude': 51.5074, 'longitude': -0.1278}, timeout=5)
if response3.status_code == 200:
    data3 = response3.json()
    print('London weather (cached):', data3.get('weather_conditions'), '- Source:', data3.get('data_source'))

print()
print('Testing prediction with extreme location differences:')
print('=' * 50)

# Test with very different locations
test_cases = [
    {'name': 'Arctic', 'lat': 78.0, 'lon': 15.0, 'expected_weather': 7},  # Snow
    {'name': 'Tropical', 'lat': 5.0, 'lon': 100.0, 'expected_weather': 3},  # Rain
    {'name': 'Desert', 'lat': 25.0, 'lon': 45.0, 'expected_weather': 1},   # Clear
]

for case in test_cases:
    try:
        data = {
            'latitude': case['lat'],
            'longitude': case['lon'],
            'weather_conditions': case['expected_weather'],
            'hour': 12,
            'speed_limit': 30
        }

        response = requests.post('http://localhost:5000/predict_risk', json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            risk_value = result.get('risk_value', 0)
            risk_level = result.get('risk_level', 'N/A')
            source = result.get('prediction_source', 'unknown')
            print(case['name'] + ': Risk ' + str(round(risk_value, 1)) + ' (' + risk_level + ') - ' + source)
        else:
            print(case['name'] + ': ERROR ' + str(response.status_code))

    except Exception as e:
        print(case['name'] + ': Exception - ' + str(e)[:30] + '...')
