import requests
import os

# Set API key
os.environ['OPENWEATHER_API_KEY'] = '282df8705ee0306c3916bb5caaad9170'

# Test different cities with extreme weather differences
cities = [
    {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
    {'name': 'Reykjavik', 'lat': 64.1466, 'lon': -21.9426},  # Very northern
    {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},   # Very tropical
]

print('Testing weather API for extreme location differences:')
print('=' * 60)

for city in cities:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={city['lat']}&lon={city['lon']}&appid={os.environ['OPENWEATHER_API_KEY']}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            weather_main = data['weather'][0]['main']
            print(city['name'] + ': ' + str(round(temp, 1)) + '°C, ' + weather_main + ', Lat: ' + str(round(city['lat'], 2)))
        else:
            print(city['name'] + ': API Error ' + str(response.status_code))
    except Exception as e:
        print(city['name'] + ': Error - ' + str(e)[:30] + '...')

print('')
print('Testing risk calculation logic:')
print('=' * 60)

# Test the risk calculation logic directly
def calculate_location_risk(lat, weather_code, hour=12):
    risk_score = 2.0

    # Geographic risk factors
    if lat > 40:  # Northern regions
        risk_score += 0.2
        geo_factor = 'Northern (+0.2)'
    elif lat < 20:  # Tropical regions
        risk_score += 0.3
        geo_factor = 'Tropical (+0.3)'
    else:
        geo_factor = 'Temperate (+0.0)'

    # Weather risk
    if weather_code in [2, 3, 5, 6]:  # Rain, Drizzle, Mist, Fog
        risk_score += 0.4
        weather_factor = 'Bad weather (+0.4)'
    elif weather_code == 7:  # Snow
        risk_score += 0.5
        weather_factor = 'Snow (+0.5)'
    else:
        weather_factor = 'Good weather (+0.0)'

    # Time risk
    if hour < 6 or hour > 22:
        risk_score += 0.3
        time_factor = 'Night (+0.3)'
    elif 7 <= hour <= 9 or 17 <= hour <= 19:
        risk_score += 0.2
        time_factor = 'Rush hour (+0.2)'
    else:
        time_factor = 'Normal time (+0.0)'

    return risk_score, geo_factor, weather_factor, time_factor

for city in cities:
    # Mock weather codes based on typical conditions
    weather_codes = {'London': 1, 'Mumbai': 3, 'Reykjavik': 7, 'Singapore': 3}
    risk, geo, weather, time_factor = calculate_location_risk(city['lat'], weather_codes.get(city['name'], 1))
    print(city['name'] + ': Risk ' + str(round(risk, 1)) + ' (' + geo + ', ' + weather + ', ' + time_factor + ')')
