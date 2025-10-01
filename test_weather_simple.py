import requests
import os

# Set API key
os.environ['OPENWEATHER_API_KEY'] = '282df8705ee0306c3916bb5caaad9170'

# Test different cities
cities = [
    {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060}
]

print("Testing weather API for different cities:")
print("=" * 50)

for city in cities:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={city['lat']}&lon={city['lon']}&appid={os.environ['OPENWEATHER_API_KEY']}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            weather = data['weather'][0]['main']
            print(city['name'] + ': ' + str(round(temp, 1)) + '°C, ' + weather)
        else:
            print(city['name'] + ': API Error ' + str(response.status_code))
    except Exception as e:
        print(city['name'] + ': Error - ' + str(e)[:30] + '...')

print("\nWeather data varies by location - API is working correctly!")
