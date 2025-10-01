# Quick Setup Guide

## 1. Clone & Install
```bash
git clone <your-repo-url>
cd road
pip install -r requirements.txt
```

## 2. Configure API Keys
```bash
# Copy example file
cp .env.example .env

# Edit .env with your API keys
OPENWEATHER_API_KEY=your_key_here
TOMTOM_API_KEY=your_key_here  # Optional
```

## 3. Run Application
```bash
python app.py
```

Visit: http://127.0.0.1:5000

## 4. Get API Keys
- **OpenWeatherMap**: https://openweathermap.org/api (Free)
- **TomTom**: https://developer.tomtom.com/ (Optional)

## Features
- ✅ Real-time accident risk prediction
- ✅ Interactive heatmaps with real data
- ✅ Global city support (including small cities)
- ✅ Live weather integration
- ✅ Multiple data sources (OSM, TomTom, Known hotspots)