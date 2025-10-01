import requests
import os

# Set API key
os.environ['OPENWEATHER_API_KEY'] = '282df8705ee0306c3916bb5caaad9170'

# Test weather API directly
cities = [
    ('London', 51.5074, -0.1278),
    ('Mumbai', 19.0760, 72.8777),
    ('Sydney', -33.8688, 151.2093),
    ('Delhi', 28.7041, 77.1025),
]

print('Weather API Test:')
print('=' * 50)

for name, lat, lon in cities:
    try:
        url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={os.environ["OPENWEATHER_API_KEY"]}&units=metric'
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            weather = data['weather'][0]['main']
            print(name + ': ' + str(round(temp, 1)) + '°C, ' + weather)
        else:
            print(name + ': Error ' + str(response.status_code))
    except Exception as e:
        print(name + ': Exception - ' + str(e)[:30] + '...')

print()
print('Flask Weather Endpoint Test:')
print('=' * 50)

for name, lat, lon in cities:
    try:
        response = requests.post('http://localhost:5000/fetch_location_data',
                               json={'latitude': lat, 'longitude': lon}, timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(name + ': Weather ' + str(data.get('weather_conditions')) + ' - ' + data.get('data_source'))
        else:
            print(name + ': Error ' + str(response.status_code))
    except Exception as e:
        print(name + ': Exception - ' + str(e)[:30] + '...')
