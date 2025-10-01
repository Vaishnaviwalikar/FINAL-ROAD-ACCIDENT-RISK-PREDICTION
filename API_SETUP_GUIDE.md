# Real-World API Setup Guide

## OpenWeatherMap API Integration

Your application now supports real-time weather data from OpenWeatherMap API! Here's how to set it up:

### Step 1: Get Your Free API Key

1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Navigate to your API keys section
4. Copy your API key

### Step 2: Set Your API Key

You have several options to set your API key:

#### Option 1: Environment Variable (Recommended)
```bash
# Windows Command Prompt
set OPENWEATHER_API_KEY=your_actual_api_key_here

# Windows PowerShell
$env:OPENWEATHER_API_KEY="your_actual_api_key_here"

# Then run your app
python app.py
```

#### Option 2: Directly in Code (For Development Only)
Replace `'demo_key'` in `app.py` with your actual API key:
```python
api_key = os.environ.get('OPENWEATHER_API_KEY', 'your_actual_api_key_here')
```

### Step 3: Test the Integration

1. Restart your Flask server
2. Click on the map
3. Check the browser console - you should see "Using real weather data" instead of "Using simulated weather data"
4. The form fields should now populate with real weather conditions for that location

### API Features

- **Real-time weather conditions**: Rain, snow, clear skies, etc.
- **Temperature data**: Current temperature at the location
- **Cloud cover**: Affects light conditions
- **Wind speed**: Can affect driving conditions
- **Weather description**: Detailed weather information

### Fallback System

If the API is unavailable or you don't have an API key set, the system will automatically fall back to simulated weather data based on location coordinates and time of day.

### Rate Limits

The free OpenWeatherMap API allows:
- 1,000 API calls per day
- 60 calls per minute

The application caches responses and makes efficient use of API calls.

### Troubleshooting

1. **"demo_key" still showing**: Make sure you set the environment variable correctly and restart the server
2. **API errors**: Check your API key is valid and has sufficient credits
3. **Slow responses**: The API call adds ~1-2 seconds to the response time

### Future Enhancements

You can also integrate:
- **Traffic APIs** (like Google Maps Traffic API)
- **Road condition APIs** (like HERE Maps or TomTom)
- **Geocoding APIs** for better road type detection
- **Historical weather data** for trend analysis