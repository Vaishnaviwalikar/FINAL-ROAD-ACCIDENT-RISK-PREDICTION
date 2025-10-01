#!/usr/bin/env python3
"""
Simplified Flask App for RoadSafe AI
Removes heavy dependencies for initial testing
"""
import os
import sys
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.debug = True

# Simple in-memory cache for demo purposes
demo_cache = {}

def get_openweather_api_key():
    """Get OpenWeatherMap API key from environment"""
    return os.environ.get('OPENWEATHER_API_KEY', 'demo_key')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index_new.html')

@app.route('/heatmap')
def heatmap_dashboard():
    """Serve the interactive risk heatmap dashboard"""
    return render_template('heatmap_dashboard.html')

@app.route('/historical-dashboard')
def historical_dashboard():
    """Serve the historical analytics dashboard"""
    return render_template('historical_dashboard.html')

@app.route('/government')
def government_dashboard():
    """Serve the government dashboard for traffic authorities"""
    return render_template('government_dashboard.html')

@app.route('/geocode_city', methods=['POST'])
def geocode_city():
    """
    Geocodes a city name to its latitude and longitude using OpenWeatherMap API.
    """
    data = request.get_json()
    city_name = data.get('city', 'London')

    # Demo response for testing
    demo_locations = {
        'london': {'lat': 51.5074, 'lon': -0.1278},
        'mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'new york': {'lat': 40.7128, 'lon': -74.0060},
        'delhi': {'lat': 28.7041, 'lon': 77.1025},
        'tokyo': {'lat': 35.6762, 'lon': 139.6503}
    }

    location = demo_locations.get(city_name.lower(), demo_locations['london'])

    return jsonify({
        'success': True,
        'latitude': location['lat'],
        'longitude': location['lon'],
        'city': city_name
    })

@app.route('/fetch_location_data', methods=['POST'])
def fetch_location_data():
    """
    Fetch weather and location data for risk prediction.
    """
    data = request.get_json()
    lat = data.get('latitude', 51.5074)
    lon = data.get('longitude', -0.1278)

    # Demo weather data
    demo_weather = {
        'temperature': 15.5,
        'humidity': 65,
        'wind_speed': 12.3,
        'weather_main': 'Clouds',
        'weather_description': 'Partly cloudy',
        'data_source': 'demo'
    }

    return jsonify({
        'success': True,
        'latitude': lat,
        'longitude': lon,
        'hour': datetime.now().hour,
        'day_of_week': datetime.now().weekday() + 1,
        'month': datetime.now().month,
        'light_conditions': 1 if 6 <= datetime.now().hour <= 18 else 0,
        'weather_conditions': 1,  # Fine
        'road_surface': 0,  # Dry
        'speed_limit': 30,
        **demo_weather
    })

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    """
    Predict accident risk using demo data.
    """
    data = request.get_json()

    # Simple demo risk calculation
    hour = data.get('hour', 12)
    weather = data.get('weather_conditions', 1)
    road_surface = data.get('road_surface', 0)

    # Demo risk factors
    time_risk = 0.3 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.1
    weather_risk = 0.2 if weather != 1 else 0.0
    surface_risk = 0.3 if road_surface != 0 else 0.0

    total_risk = min(time_risk + weather_risk + surface_risk, 1.0)

    if total_risk < 0.33:
        risk_level = "Low"
    elif total_risk < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return jsonify({
        'success': True,
        'risk_level': risk_level,
        'risk_value': round(total_risk, 3),
        'confidence': 0.75,
        'prediction_source': 'demo_model',
        'contributing_factors': ['Demo calculation', 'Time-based risk', 'Weather factors']
    })

# API Routes for Traffic (Demo)
@app.route('/api/traffic/providers', methods=['GET'])
def get_traffic_providers():
    """Get available traffic data providers"""
    return jsonify({
        'success': True,
        'providers': [
            {
                'name': 'demo',
                'display_name': 'Demo',
                'available': True,
                'features': ['simulated data', 'always available'],
                'coverage': 'global',
                'note': 'Demo data based on typical patterns'
            }
        ],
        'default_provider': 'demo'
    })

@app.route('/api/traffic/current', methods=['GET'])
def get_current_traffic():
    """Get current traffic conditions (demo)"""
    lat = float(request.args.get('lat', 51.5074))
    lon = float(request.args.get('lon', -0.1278))

    # Demo traffic data
    hour = datetime.now().hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        congestion = 0.7
        speed = 25
    else:
        congestion = 0.2
        speed = 55

    return jsonify({
        'success': True,
        'location': {'lat': lat, 'lon': lon},
        'traffic': {
            'congestion_level': congestion,
            'congestion_percentage': round(congestion * 100, 1),
            'average_speed_kmh': speed,
            'free_flow_speed_kmh': 60,
            'confidence': 0.6,
            'provider': 'demo',
            'incidents_count': 0
        },
        'risk_analysis': {
            'traffic_risk_factor': congestion * 0.5,
            'risk_level': 'Low' if congestion < 0.5 else 'Medium',
            'contributing_factors': ['Demo traffic data']
        }
    })

# API Routes for Heatmap (Demo)
@app.route('/api/heatmap/preview', methods=['GET'])
def get_heatmap_preview():
    """Get a quick preview heatmap for testing"""
    lat = float(request.args.get('lat', 51.5074))
    lon = float(request.args.get('lon', -0.1278))

    # Generate demo heatmap points
    points = []
    for i in range(10):
        for j in range(10):
            point_lat = lat + (i - 5) * 0.01
            point_lon = lon + (j - 5) * 0.01
            risk = 0.1 + (i * j) / 1000  # Simple demo risk calculation
            points.append({
                'lat': point_lat,
                'lon': point_lon,
                'risk_level': round(risk, 3),
                'risk_category': 'low' if risk < 0.3 else 'medium' if risk < 0.6 else 'high'
            })

    return jsonify({
        'success': True,
        'heatmap': {
            'center_lat': lat,
            'center_lon': lon,
            'points_count': len(points),
            'preview_points': points,
            'statistics': {
                'total_points': len(points),
                'avg_risk': 0.25,
                'high_risk_areas': 5,
                'risk_distribution': {'low': 70, 'medium': 25, 'high': 5}
            }
        }
    })

# API Routes for Historical Data (Demo)
@app.route('/api/historical/statistics', methods=['GET'])
def get_historical_statistics():
    """Get historical accident statistics (demo)"""
    lat = float(request.args.get('lat', 51.5074))
    lon = float(request.args.get('lon', -0.1278))

    return jsonify({
        'success': True,
        'location': {'lat': lat, 'lon': lon},
        'statistics': {
            'total_accidents': 45,
            'average_severity': 2.1,
            'peak_hours': [8, 17, 18],
            'high_risk_factors': ['Rush hour', 'Urban area', 'Intersections'],
            'risk_trends': 'Increasing during winter months'
        }
    })

@app.route('/api/historical/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """Get comprehensive dashboard data (demo)"""
    return jsonify({
        'success': True,
        'summary': {
            'total_records': 1000,
            'date_range': '2023-01-01 to 2024-12-31',
            'avg_daily_accidents': 2.7
        },
        'risk_metrics': {
            'overall_risk': 0.35,
            'trend': 'stable',
            'confidence': 0.85
        },
        'charts_data': {
            'hourly_distribution': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 95, 90, 85, 80, 75],
            'weather_impact': [0.2, 0.5, 0.8, 0.3],
            'road_type_risk': [0.4, 0.6, 0.3, 0.7]
        }
    })

if __name__ == '__main__':
    print("🚀 Starting RoadSafe AI (Demo Mode)")
    print("=" * 50)
    print("✅ All dependencies loaded successfully")
    print("✅ Demo services active")
    print("✅ Web interface ready")
    print("=" * 50)
    print("🌐 Access the application at: http://localhost:5000")
    print("=" * 50)

    # Set environment variables
    os.environ['OPENWEATHER_API_KEY'] = get_openweather_api_key()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
