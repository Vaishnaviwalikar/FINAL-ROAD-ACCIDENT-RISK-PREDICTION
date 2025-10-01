import requests
import time

print('Testing complete city flow: Geocode -> Weather -> Predict')
print('=' * 60)

# Test different cities
cities = ['London', 'Mumbai', 'Sydney', 'Delhi']

for city in cities:
    print(f'\n--- Testing {city} ---')

    try:
        # Step 1: Geocode the city
        geo_response = requests.post('http://localhost:5000/geocode_city',
                                   json={'city': city}, timeout=5)
        if geo_response.status_code == 200:
            geo_data = geo_response.json()
            lat, lon = geo_data['latitude'], geo_data['longitude']
            print(f'  Geocode: {lat:.2f}, {lon:.2f}')
        else:
            print(f'  Geocode ERROR: {geo_response.status_code}')
            continue

        # Small delay to avoid rate limiting
        time.sleep(1)

        # Step 2: Fetch weather data
        weather_response = requests.post('http://localhost:5000/fetch_location_data',
                                       json={'latitude': lat, 'longitude': lon}, timeout=5)
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            weather_code = weather_data.get('weather_conditions')
            data_source = weather_data.get('data_source')
            print(f'  Weather: {weather_code} ({data_source})')
        else:
            print(f'  Weather ERROR: {weather_response.status_code}')
            continue

        # Step 3: Get prediction
        pred_data = {
            'latitude': lat,
            'longitude': lon,
            'weather_conditions': weather_code,
            'hour': 12,
            'speed_limit': 30
        }

        pred_response = requests.post('http://localhost:5000/predict_risk',
                                    json=pred_data, timeout=5)
        if pred_response.status_code == 200:
            pred_result = pred_response.json()
            risk_value = pred_result.get('risk_value')
            risk_level = pred_result.get('risk_level')
            source = pred_result.get('prediction_source')
            print(f'  Prediction: {risk_level} ({risk_value:.1f}) - {source}')
        else:
            print(f'  Prediction ERROR: {pred_response.status_code}')

    except Exception as e:
        print(f'  Exception: {str(e)[:50]}...')

print('\n' + '=' * 60)
print('If all cities show different risk values above, the backend is working.')
print('The issue might be in the frontend integration.')
print('Check if your frontend is:')
print('1. Properly calling /geocode_city for each city')
print('2. Fetching fresh weather data for each location')
print('3. Passing correct weather data to /predict_risk')
