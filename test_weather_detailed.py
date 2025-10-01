import requests
import os

# Set API key
os.environ['OPENWEATHER_API_KEY'] = '282df8705ee0306c3916bb5caaad9170'

# Test weather API directly for different cities
cities = [
    {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
    {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093},
    {'name': 'Delhi', 'lat': 28.7041, 'lon': 77.1025},
    {'name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
]

print('Testing weather API directly for different cities:')
print('=' * 60)

for city in cities:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={city['lat']}&lon={city['lon']}&appid={os.environ['OPENWEATHER_API_KEY']}&units=metric"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            weather_main = data['weather'][0]['main']
            weather_desc = data['weather'][0]['description']

            # Map weather to our codes
            weather_code = 1  # Default clear
            if 'rain' in weather_main.lower() or 'drizzle' in weather_main.lower():
                weather_code = 2
            elif 'snow' in weather_main.lower():
                weather_code = 7
            elif 'thunder' in weather_main.lower():
                weather_code = 4
            elif 'fog' in weather_main.lower() or 'mist' in weather_main.lower():
                weather_code = 5
            elif 'cloud' in weather_main.lower():
                weather_code = 3

            print(f"{city['name']"8"}: {temp"5.1f"}°C, {weather_main} ({weather_code}) - {weather_desc}")
        else:
            print(f"{city['name']"8"}: API Error {response.status_code}")

    except Exception as e:
        print(f"{city['name']"8"}: Exception - {str(e)[:40]}...")

print()
print('Testing Flask weather endpoint:')
print('=' * 60)

# Test Flask weather endpoint with same coordinates
for city in cities[:4]:  # Test first 4 cities
    try:
        response = requests.post('http://localhost:5000/fetch_location_data',
                               json={'latitude': city['lat'], 'longitude': city['lon']}, timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"{city['name']"8"}: Weather {data.get('weather_conditions')} - {data.get('data_source')} - {data.get('main_weather')}")
        else:
            print(f"{city['name']"8"}: Flask Error {response.status_code}")

    except Exception as e:
        print(f"{city['name']"8"}: Exception - {str(e)[:40]}...")
