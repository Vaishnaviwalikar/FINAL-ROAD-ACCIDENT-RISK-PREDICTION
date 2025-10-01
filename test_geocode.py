import requests

# Test the geocode_city endpoint directly
print('Testing /geocode_city endpoint:')
print('=' * 40)

test_cities = ['London', 'Mumbai', 'Sydney', 'Delhi', 'Tokyo']

for city in test_cities:
    try:
        response = requests.post('http://localhost:5000/geocode_city',
                               json={'city': city}, timeout=5)
        print(city + ': Status ' + str(response.status_code))
        if response.status_code == 200:
            data = response.json()
            print('  -> ' + data.get('city') + ' (' + str(data.get('latitude')) + ', ' + str(data.get('longitude')) + ')')
        else:
            print('  -> Error: ' + response.text[:100] + '...')
    except Exception as e:
        print(city + ': Exception - ' + str(e)[:50] + '...')

print()
print('Testing /fetch_location_data endpoint:')
print('=' * 40)

# Test weather fetch for different locations
for city in test_cities[:3]:  # Test first 3 cities
    try:
        # First get coordinates
        geo_response = requests.post('http://localhost:5000/geocode_city',
                                   json={'city': city}, timeout=5)
        if geo_response.status_code == 200:
            geo_data = geo_response.json()
            lat, lon = geo_data['latitude'], geo_data['longitude']

            # Then fetch weather
            weather_response = requests.post('http://localhost:5000/fetch_location_data',
                                           json={'latitude': lat, 'longitude': lon}, timeout=5)
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                print(city + ': Weather ' + str(weather_data.get('weather_conditions')) + ' - ' + weather_data.get('data_source'))
            else:
                print(city + ': Weather API Error ' + str(weather_response.status_code))
    except Exception as e:
        print(city + ': Error - ' + str(e)[:50] + '...')
