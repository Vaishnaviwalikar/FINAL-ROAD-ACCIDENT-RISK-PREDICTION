"""
API Configuration for Road Traffic Accident Prediction App
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from a local .env file if present
load_dotenv()

def get_openweather_api_key() -> Optional[str]:
    """
    Get OpenWeatherMap API key from environment variables.
    Always use real API - no demo mode.
    """
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is required")
    return api_key

def is_demo_mode() -> bool:
    """Always use real API data - no demo mode"""
    return False

# Weather condition mappings for the ML model
WEATHER_CONDITIONS_MAP = {
    'Clear': 1,         # Fine weather
    'Clouds': 1,        # Fine weather (light clouds)
    'Rain': 2,          # Raining
    'Drizzle': 2,       # Light rain
    'Thunderstorm': 2,  # Heavy rain
    'Snow': 3,          # Snowing
    'Sleet': 3,         # Snow/ice
    'Fog': 7,           # Fog/mist
    'Mist': 7,          # Fog/mist
    'Haze': 7,          # Fog/mist
    'Smoke': 7,         # Fog/mist
    'Dust': 7,          # Fog/mist
    'Sand': 7,          # Fog/mist
    'Ash': 7,           # Fog/mist
    'Squall': 2,        # Windy rain
    'Tornado': 2        # Extreme weather
}

# Road surface conditions based on weather
def get_road_surface_from_weather(weather_main: str, temperature: float) -> int:
    """
    Determine road surface condition based on weather and temperature
    0: Dry, 1: Wet, 2: Snow, 3: Frost/Ice
    """
    if weather_main in ['Rain', 'Drizzle', 'Thunderstorm']:
        return 1  # Wet
    elif weather_main in ['Snow', 'Sleet']:
        return 2 if temperature > 0 else 3  # Snow or Frost/Ice
    elif temperature <= 2:  # Near freezing
        return 3  # Frost/Ice
    else:
        return 0  # Dry

# Light conditions based on time of day
def get_light_conditions(hour: int) -> int:
    """
    Determine light conditions based on hour
    0: Dark-lit, 1: Dark-unlit, 2: Daylight, 3: Dark-no lighting
    """
    if 6 <= hour <= 18:
        return 2  # Daylight
    elif 19 <= hour <= 23 or 5 <= hour <= 6:
        return 0  # Dark-lit (street lighting)
    else:
        return 1  # Dark-unlit (late night/early morning)
