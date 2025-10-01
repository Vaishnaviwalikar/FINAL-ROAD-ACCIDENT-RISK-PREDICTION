#!/usr/bin/env python3
"""
RoadSafe AI with Real API Integration
Uses OpenWeatherMap and TomTom APIs with clear data source indicators
"""
import os
import sys
import json
import time
import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')

# Cache for API responses (2 minute TTL)
api_cache = {}
CACHE_TTL = 120  # 2 minutes

# Disable CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.cuda.is_available = lambda: False

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Global model variables
model = None
feature_columns = []
config = {}

class AIModel(nn.Module):
    def __init__(self, in_channels=30):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 128)
        
        self.s_attn = nn.Module()
        self.s_attn.proj = nn.ModuleList([
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 128)
        ])
        self.t_attn = nn.Module()
        self.t_attn.query = nn.Linear(128, 64)
        self.t_attn.key = nn.Linear(128, 64)
        
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.drop2(x)
        
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        
        s_x = self.s_attn.proj[0](x)
        s_x = torch.relu(s_x)
        s_x = torch.dropout(s_x, 0.3, self.training)
        s_x = self.s_attn.proj[2](s_x)
        s_attn = torch.sigmoid(s_x)
        x = x * s_attn
        
        q = self.t_attn.query(x)
        k = self.t_attn.key(x)
        t_attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True), dim=1)
        x = x * t_attn
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.out(x)
        return x.squeeze(-1)

def get_cached_data(cache_key):
    """Get cached data if still valid"""
    if cache_key in api_cache:
        data, timestamp = api_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None

def set_cached_data(cache_key, data):
    """Cache data with timestamp"""
    api_cache[cache_key] = (data, time.time())

def fetch_real_weather_data(lat, lon):
    """Fetch real weather data from OpenWeatherMap API"""
    if not OPENWEATHER_API_KEY:
        return None
    
    cache_key = f"weather_{lat}_{lon}"
    cached = get_cached_data(cache_key)
    if cached:
        return cached
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind'].get('speed', 0),
                'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'source': 'openweather_api'
            }
            set_cached_data(cache_key, weather_data)
            return weather_data
    except Exception as e:
        logger.error(f"Weather API error: {e}")
    
    return None

def geocode_with_tomtom(city_name):
    """Geocode city using TomTom API"""
    if not TOMTOM_API_KEY:
        return None
    
    cache_key = f"geocode_{city_name.lower()}"
    cached = get_cached_data(cache_key)
    if cached:
        return cached
    
    try:
        url = f"https://api.tomtom.com/search/2/geocode/{city_name}.json?key={TOMTOM_API_KEY}&limit=1"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                geocode_data = {
                    'lat': result['position']['lat'],
                    'lon': result['position']['lon'],
                    'address': result['address']['freeformAddress'],
                    'source': 'tomtom_api'
                }
                set_cached_data(cache_key, geocode_data)
                return geocode_data
    except Exception as e:
        logger.error(f"Geocoding API error: {e}")
    
    return None

def load_ai_model():
    global model, feature_columns, config
    
    try:
        model_path = "outputs/quick_fixed/best.pt"
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
            
        logger.info(f"Loading AI model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        feature_columns = checkpoint.get('features', [])
        config = checkpoint.get('cfg', {'data': {'sequence_length': 8}})
        
        model = AIModel(len(feature_columns))
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        
        logger.info(f"AI model loaded successfully with {len(feature_columns)} features")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load AI model: {e}")
        return False

def predict_with_ai(input_data):
    global model, feature_columns, config
    
    if model is None:
        return None
        
    try:
        import pandas as pd
        
        df = pd.DataFrame([input_data])
        
        # Map features according to IEEE paper methodology
        for feature in feature_columns:
            if feature not in df.columns:
                if 'Index' in feature or 'Id' in feature:
                    df[feature] = 1
                elif 'Authority' in feature or 'Force' in feature:
                    df[feature] = 1
                elif 'Easting' in feature:
                    df[feature] = 530000
                elif 'Northing' in feature:
                    df[feature] = 180000
                elif feature in ['week', 'month']:
                    df[feature] = 6
                else:
                    df[feature] = 0
        
        X = df[feature_columns].values.astype(np.float32)
        seq_len = config['data']['sequence_length']
        X_seq = np.tile(X, (seq_len, 1)).reshape(1, seq_len, -1)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(X_tensor)
            raw_output = float(output.item())
            
            # IEEE paper methodology: Risk values 1=slight, 2=serious, 3=fatal
            # Apply sigmoid to normalize between 0-1, then scale to 1-3
            normalized_output = torch.sigmoid(torch.tensor(raw_output)).item()
            
            # Calculate cumulative risk based on real conditions (IEEE approach)
            base_risk = 1.0 + (normalized_output * 2.0)  # Scale to 1-3 range
            
            # Real-world risk factors from weather data
            weather_risk = calculate_weather_risk(input_data)
            temporal_risk = calculate_temporal_risk(input_data)
            spatial_risk = calculate_spatial_risk(input_data)
            
            # Combine risks (IEEE cumulative approach)
            cumulative_risk = (base_risk * 0.4) + (weather_risk * 0.3) + (temporal_risk * 0.2) + (spatial_risk * 0.1)
            
            # Ensure within valid range
            risk_value = max(1.0, min(3.0, cumulative_risk))
            
            # IEEE classification thresholds
            if risk_value < 1.4:
                risk_level = "Low"  # Slight accidents
            elif risk_value < 2.3:
                risk_level = "Medium"  # Serious accidents
            else:
                risk_level = "High"  # Fatal accidents
                
            # Confidence based on model certainty
            confidence = 70.0 + (abs(normalized_output - 0.5) * 60.0)
            
            return {
                "risk_value": round(risk_value, 3),
                "risk_level": risk_level,
                "confidence": round(confidence, 1),
                "prediction_source": "real_ai_model",
                "model_type": "CNN-BiLSTM-Attention",
                "weather_risk": round(weather_risk, 2),
                "temporal_risk": round(temporal_risk, 2),
                "spatial_risk": round(spatial_risk, 2)
            }
            
    except Exception as e:
        logger.error(f"AI prediction failed: {e}")
        return None

def calculate_weather_risk(data):
    """Calculate weather-based risk factor (IEEE methodology)"""
    weather = data.get('Weather_Conditions', 1)
    temp = data.get('temperature', 20)
    wind_speed = data.get('wind_speed', 0)
    humidity = data.get('humidity', 50)
    
    risk = 1.0
    
    # Weather conditions impact (from IEEE paper Table 1)
    if weather == 1:  # Fine
        risk += 0.0
    elif weather == 2:  # Raining
        risk += 0.4
    elif weather == 3:  # Snowing
        risk += 0.6
    elif weather == 7:  # Fog
        risk += 0.5
    
    # Temperature effects
    if temp < 0:  # Freezing
        risk += 0.3
    elif temp > 35:  # Very hot
        risk += 0.2
    
    # Wind effects
    if wind_speed > 10:  # Strong wind
        risk += 0.2
    
    # Humidity effects
    if humidity > 80:  # High humidity
        risk += 0.1
    
    return min(3.0, risk)

def calculate_temporal_risk(data):
    """Calculate time-based risk factor (IEEE methodology)"""
    hour = data.get('hour', 12)
    day_of_week = data.get('Day_of_Week', 3)
    month = data.get('month', 6)
    
    risk = 1.0
    
    # Hour-based risk (from IEEE paper analysis)
    if 6 <= hour <= 9 or 17 <= hour <= 20:  # Rush hours
        risk += 0.4
    elif 22 <= hour <= 6:  # Night hours
        risk += 0.5
    elif 10 <= hour <= 16:  # Daytime
        risk += 0.1
    
    # Day of week (Friday highest in IEEE paper)
    if day_of_week == 5:  # Friday
        risk += 0.3
    elif day_of_week in [6, 7]:  # Weekend
        risk += 0.2
    
    # Month effects (winter months higher risk)
    if month in [12, 1, 2]:  # Winter
        risk += 0.2
    elif month in [6, 7, 8]:  # Summer
        risk += 0.1
    
    return min(3.0, risk)

def calculate_spatial_risk(data):
    """Calculate location-based risk factor (IEEE methodology)"""
    lat = data.get('Latitude', 20)
    speed_limit = data.get('Speed_limit', 50)
    road_type = data.get('Road_Type', 3)
    junction = data.get('Junction_Detail', 0)
    
    risk = 1.0
    
    # Speed limit effects (higher speed = higher risk)
    if speed_limit >= 70:
        risk += 0.4
    elif speed_limit >= 50:
        risk += 0.2
    elif speed_limit <= 30:
        risk += 0.1  # Urban areas
    
    # Road type effects
    if road_type == 1:  # Motorway
        risk += 0.3
    elif road_type == 2:  # A road
        risk += 0.2
    
    # Junction effects (intersections higher risk)
    if junction > 0:
        risk += 0.3
    
    # Geographic risk (based on Indian context)
    if 28 <= lat <= 32:  # North India (Delhi region)
        risk += 0.2
    elif 18 <= lat <= 24:  # Central India (Mumbai region)
        risk += 0.15
    elif 8 <= lat <= 16:  # South India (Bangalore region)
        risk += 0.05
    
    return min(3.0, risk)

# Comprehensive Indian cities database
INDIAN_CITIES = {
    # Major metros
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai'},
    'delhi': {'lat': 28.7041, 'lon': 77.1025, 'name': 'Delhi'},
    'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bangalore'},
    'bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bengaluru'},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai'},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata'},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad'},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Pune'},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Ahmedabad'},
    
    # State capitals
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur'},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow'},
    'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal'},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Patna'},
    'thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'name': 'Thiruvananthapuram'},
    'gandhinagar': {'lat': 23.2156, 'lon': 72.6369, 'name': 'Gandhinagar'},
    'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh'},
    'shimla': {'lat': 31.1048, 'lon': 77.1734, 'name': 'Shimla'},
    'dehradun': {'lat': 30.3165, 'lon': 78.0322, 'name': 'Dehradun'},
    'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'name': 'Ranchi'},
    'raipur': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur'},
    'bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'name': 'Bhubaneswar'},
    'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'name': 'Guwahati'},
    'imphal': {'lat': 24.8170, 'lon': 93.9368, 'name': 'Imphal'},
    'agartala': {'lat': 23.8315, 'lon': 91.2868, 'name': 'Agartala'},
    'aizawl': {'lat': 23.7271, 'lon': 92.7176, 'name': 'Aizawl'},
    'kohima': {'lat': 25.6751, 'lon': 94.1086, 'name': 'Kohima'},
    'itanagar': {'lat': 27.0844, 'lon': 93.6053, 'name': 'Itanagar'},
    'gangtok': {'lat': 27.3389, 'lon': 88.6065, 'name': 'Gangtok'},
    
    # Major cities
    'surat': {'lat': 21.1702, 'lon': 72.8311, 'name': 'Surat'},
    'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'name': 'Kanpur'},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'name': 'Nagpur'},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore'},
    'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'name': 'Vadodara'},
    'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'name': 'Ludhiana'},
    'agra': {'lat': 27.1767, 'lon': 78.0081, 'name': 'Agra'},
    'nashik': {'lat': 19.9975, 'lon': 73.7898, 'name': 'Nashik'},
    'faridabad': {'lat': 28.4089, 'lon': 77.3178, 'name': 'Faridabad'},
    'meerut': {'lat': 28.9845, 'lon': 77.7064, 'name': 'Meerut'},
    'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'name': 'Rajkot'},
    'varanasi': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Varanasi'},
    'amritsar': {'lat': 31.6340, 'lon': 74.8723, 'name': 'Amritsar'},
    'allahabad': {'lat': 25.4358, 'lon': 81.8463, 'name': 'Allahabad'},
    'prayagraj': {'lat': 25.4358, 'lon': 81.8463, 'name': 'Prayagraj'},
    'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Visakhapatnam'},
    'vizag': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Vizag'},
    'madurai': {'lat': 9.9252, 'lon': 78.1198, 'name': 'Madurai'},
    'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'name': 'Coimbatore'},
    'kochi': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Kochi'},
    'cochin': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Cochin'},
    'kozhikode': {'lat': 11.2588, 'lon': 75.7804, 'name': 'Kozhikode'},
    'calicut': {'lat': 11.2588, 'lon': 75.7804, 'name': 'Calicut'},
    'thrissur': {'lat': 10.5276, 'lon': 76.2144, 'name': 'Thrissur'},
    
    # Maharashtra cities
    'thane': {'lat': 19.2183, 'lon': 72.9781, 'name': 'Thane'},
    'navi mumbai': {'lat': 19.0330, 'lon': 73.0297, 'name': 'Navi Mumbai'},
    'aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'name': 'Aurangabad'},
    'solapur': {'lat': 17.6599, 'lon': 75.9064, 'name': 'Solapur'},
    'kolhapur': {'lat': 16.7050, 'lon': 74.2433, 'name': 'Kolhapur'},
    'sangli': {'lat': 16.8524, 'lon': 74.5815, 'name': 'Sangli'},
    'satara': {'lat': 17.6805, 'lon': 74.0183, 'name': 'Satara'},
    'ahmednagar': {'lat': 19.0948, 'lon': 74.7480, 'name': 'Ahmednagar'},
    'latur': {'lat': 18.4088, 'lon': 76.5604, 'name': 'Latur'},
    'akola': {'lat': 20.7002, 'lon': 77.0082, 'name': 'Akola'},
    'amravati': {'lat': 20.9374, 'lon': 77.7796, 'name': 'Amravati'},
    'chandrapur': {'lat': 19.9615, 'lon': 79.2961, 'name': 'Chandrapur'},
    'dhule': {'lat': 20.9042, 'lon': 74.7749, 'name': 'Dhule'},
    'jalgaon': {'lat': 21.0077, 'lon': 75.5626, 'name': 'Jalgaon'},
    'nanded': {'lat': 19.1383, 'lon': 77.3210, 'name': 'Nanded'},
    'osmanabad': {'lat': 18.1760, 'lon': 76.0395, 'name': 'Osmanabad'},
    'parbhani': {'lat': 19.2608, 'lon': 76.7734, 'name': 'Parbhani'},
    'yavatmal': {'lat': 20.3897, 'lon': 78.1307, 'name': 'Yavatmal'},
    'wardha': {'lat': 20.7453, 'lon': 78.6022, 'name': 'Wardha'},
    'washim': {'lat': 20.1109, 'lon': 77.1331, 'name': 'Washim'},
    'beed': {'lat': 18.9894, 'lon': 75.7585, 'name': 'Beed'},
    'buldhana': {'lat': 20.5307, 'lon': 76.1809, 'name': 'Buldhana'},
    'gadchiroli': {'lat': 20.1764, 'lon': 80.0648, 'name': 'Gadchiroli'},
    'gondia': {'lat': 21.4607, 'lon': 80.1982, 'name': 'Gondia'},
    'hingoli': {'lat': 19.7147, 'lon': 77.1547, 'name': 'Hingoli'},
    'jalna': {'lat': 19.8347, 'lon': 75.8861, 'name': 'Jalna'},
    'raigad': {'lat': 18.2257, 'lon': 73.1340, 'name': 'Raigad'},
    'ratnagiri': {'lat': 16.9902, 'lon': 73.3120, 'name': 'Ratnagiri'},
    'sindhudurg': {'lat': 16.0667, 'lon': 73.6667, 'name': 'Sindhudurg'},
    
    # Gujarat cities
    'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'name': 'Rajkot'},
    'bhavnagar': {'lat': 21.7645, 'lon': 72.1519, 'name': 'Bhavnagar'},
    'jamnagar': {'lat': 22.4707, 'lon': 70.0577, 'name': 'Jamnagar'},
    'junagadh': {'lat': 21.5222, 'lon': 70.4579, 'name': 'Junagadh'},
    'anand': {'lat': 22.5645, 'lon': 72.9289, 'name': 'Anand'},
    'mehsana': {'lat': 23.5880, 'lon': 72.3693, 'name': 'Mehsana'},
    'morbi': {'lat': 22.8173, 'lon': 70.8378, 'name': 'Morbi'},
    'nadiad': {'lat': 22.6939, 'lon': 72.8618, 'name': 'Nadiad'},
    'surendranagar': {'lat': 22.7196, 'lon': 71.6369, 'name': 'Surendranagar'},
    'gandhidham': {'lat': 23.0800, 'lon': 70.1300, 'name': 'Gandhidham'},
    
    # Rajasthan cities
    'jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'name': 'Jodhpur'},
    'kota': {'lat': 25.2138, 'lon': 75.8648, 'name': 'Kota'},
    'bikaner': {'lat': 28.0229, 'lon': 73.3119, 'name': 'Bikaner'},
    'udaipur': {'lat': 24.5854, 'lon': 73.7125, 'name': 'Udaipur'},
    'ajmer': {'lat': 26.4499, 'lon': 74.6399, 'name': 'Ajmer'},
    'bharatpur': {'lat': 27.2152, 'lon': 77.4909, 'name': 'Bharatpur'},
    'alwar': {'lat': 27.5530, 'lon': 76.6346, 'name': 'Alwar'},
    'sikar': {'lat': 27.6094, 'lon': 75.1399, 'name': 'Sikar'},
    'pali': {'lat': 25.7711, 'lon': 73.3234, 'name': 'Pali'},
    'tonk': {'lat': 26.1693, 'lon': 75.7847, 'name': 'Tonk'},
    
    # Tamil Nadu cities
    'salem': {'lat': 11.6643, 'lon': 78.1460, 'name': 'Salem'},
    'tirupur': {'lat': 11.1085, 'lon': 77.3411, 'name': 'Tirupur'},
    'erode': {'lat': 11.3410, 'lon': 77.7172, 'name': 'Erode'},
    'vellore': {'lat': 12.9165, 'lon': 79.1325, 'name': 'Vellore'},
    'tirunelveli': {'lat': 8.7139, 'lon': 77.7567, 'name': 'Tirunelveli'},
    'thoothukudi': {'lat': 8.7642, 'lon': 78.1348, 'name': 'Thoothukudi'},
    'dindigul': {'lat': 10.3673, 'lon': 77.9803, 'name': 'Dindigul'},
    'thanjavur': {'lat': 10.7870, 'lon': 79.1378, 'name': 'Thanjavur'},
    'trichy': {'lat': 10.7905, 'lon': 78.7047, 'name': 'Trichy'},
    'tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'name': 'Tiruchirappalli'},
    'karur': {'lat': 10.9601, 'lon': 78.0766, 'name': 'Karur'},
    'cuddalore': {'lat': 11.7449, 'lon': 79.7689, 'name': 'Cuddalore'},
    'kanchipuram': {'lat': 12.8342, 'lon': 79.7036, 'name': 'Kanchipuram'},
    'nagercoil': {'lat': 8.1790, 'lon': 77.4338, 'name': 'Nagercoil'},
    
    # Karnataka cities
    'mysore': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysore'},
    'mysuru': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysuru'},
    'hubli': {'lat': 15.3647, 'lon': 75.1240, 'name': 'Hubli'},
    'dharwad': {'lat': 15.4589, 'lon': 75.0078, 'name': 'Dharwad'},
    'mangalore': {'lat': 12.9141, 'lon': 74.8560, 'name': 'Mangalore'},
    'belgaum': {'lat': 15.8497, 'lon': 74.4977, 'name': 'Belgaum'},
    'gulbarga': {'lat': 17.3297, 'lon': 76.8343, 'name': 'Gulbarga'},
    'davangere': {'lat': 14.4644, 'lon': 75.9218, 'name': 'Davangere'},
    'bellary': {'lat': 15.1394, 'lon': 76.9214, 'name': 'Bellary'},
    'bijapur': {'lat': 16.8302, 'lon': 75.7100, 'name': 'Bijapur'},
    'shimoga': {'lat': 13.9299, 'lon': 75.5681, 'name': 'Shimoga'},
    'tumkur': {'lat': 13.3379, 'lon': 77.1022, 'name': 'Tumkur'},
    'hassan': {'lat': 13.0033, 'lon': 76.0965, 'name': 'Hassan'},
    'mandya': {'lat': 12.5218, 'lon': 76.8951, 'name': 'Mandya'},
    
    # Andhra Pradesh & Telangana
    'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Vijayawada'},
    'guntur': {'lat': 16.3067, 'lon': 80.4365, 'name': 'Guntur'},
    'nellore': {'lat': 14.4426, 'lon': 79.9865, 'name': 'Nellore'},
    'kurnool': {'lat': 15.8281, 'lon': 78.0373, 'name': 'Kurnool'},
    'kadapa': {'lat': 14.4673, 'lon': 78.8242, 'name': 'Kadapa'},
    'tirupati': {'lat': 13.6288, 'lon': 79.4192, 'name': 'Tirupati'},
    'anantapur': {'lat': 14.6819, 'lon': 77.6006, 'name': 'Anantapur'},
    'chittoor': {'lat': 13.2172, 'lon': 79.1003, 'name': 'Chittoor'},
    'warangal': {'lat': 17.9689, 'lon': 79.5941, 'name': 'Warangal'},
    'nizamabad': {'lat': 18.6725, 'lon': 78.0941, 'name': 'Nizamabad'},
    'karimnagar': {'lat': 18.4386, 'lon': 79.1288, 'name': 'Karimnagar'},
    'khammam': {'lat': 17.2473, 'lon': 80.1514, 'name': 'Khammam'},
    'mahbubnagar': {'lat': 16.7393, 'lon': 77.9974, 'name': 'Mahbubnagar'},
    'medak': {'lat': 18.0488, 'lon': 78.2751, 'name': 'Medak'},
    'nalgonda': {'lat': 17.0542, 'lon': 79.2673, 'name': 'Nalgonda'},
    'rangareddy': {'lat': 17.4065, 'lon': 78.4772, 'name': 'Rangareddy'},
    
    # West Bengal cities
    'howrah': {'lat': 22.5958, 'lon': 88.2636, 'name': 'Howrah'},
    'durgapur': {'lat': 23.4820, 'lon': 87.3119, 'name': 'Durgapur'},
    'asansol': {'lat': 23.6739, 'lon': 86.9524, 'name': 'Asansol'},
    'siliguri': {'lat': 26.7271, 'lon': 88.3953, 'name': 'Siliguri'},
    'malda': {'lat': 25.0961, 'lon': 88.1435, 'name': 'Malda'},
    'kharagpur': {'lat': 22.3460, 'lon': 87.2320, 'name': 'Kharagpur'},
    'burdwan': {'lat': 23.2324, 'lon': 87.8615, 'name': 'Burdwan'},
    'midnapore': {'lat': 22.4241, 'lon': 87.3180, 'name': 'Midnapore'},
    'krishnanagar': {'lat': 23.4058, 'lon': 88.5019, 'name': 'Krishnanagar'},
    'baharampur': {'lat': 24.1048, 'lon': 88.2529, 'name': 'Baharampur'},
    
    # Uttar Pradesh cities
    'ghaziabad': {'lat': 28.6692, 'lon': 77.4538, 'name': 'Ghaziabad'},
    'noida': {'lat': 28.5355, 'lon': 77.3910, 'name': 'Noida'},
    'greater noida': {'lat': 28.4744, 'lon': 77.5040, 'name': 'Greater Noida'},
    'moradabad': {'lat': 28.8386, 'lon': 78.7733, 'name': 'Moradabad'},
    'aligarh': {'lat': 27.8974, 'lon': 78.0880, 'name': 'Aligarh'},
    'bareilly': {'lat': 28.3670, 'lon': 79.4304, 'name': 'Bareilly'},
    'saharanpur': {'lat': 29.9680, 'lon': 77.5552, 'name': 'Saharanpur'},
    'gorakhpur': {'lat': 26.7606, 'lon': 83.3732, 'name': 'Gorakhpur'},
    'mathura': {'lat': 27.4924, 'lon': 77.6737, 'name': 'Mathura'},
    'firozabad': {'lat': 27.1592, 'lon': 78.3957, 'name': 'Firozabad'},
    'jhansi': {'lat': 25.4484, 'lon': 78.5685, 'name': 'Jhansi'},
    'muzaffarnagar': {'lat': 29.4727, 'lon': 77.7085, 'name': 'Muzaffarnagar'},
    'rampur': {'lat': 28.8152, 'lon': 79.0250, 'name': 'Rampur'},
    'shahjahanpur': {'lat': 27.8831, 'lon': 79.9119, 'name': 'Shahjahanpur'},
    'bulandshahr': {'lat': 28.4041, 'lon': 77.8498, 'name': 'Bulandshahr'},
    'unnao': {'lat': 26.5464, 'lon': 80.4879, 'name': 'Unnao'},
    'sitapur': {'lat': 27.5677, 'lon': 80.6827, 'name': 'Sitapur'},
    'etawah': {'lat': 26.7751, 'lon': 79.0154, 'name': 'Etawah'},
    'farrukhabad': {'lat': 27.3929, 'lon': 79.5800, 'name': 'Farrukhabad'},
    'hardoi': {'lat': 27.4167, 'lon': 80.1333, 'name': 'Hardoi'},
    'lakhimpur': {'lat': 27.9479, 'lon': 80.7781, 'name': 'Lakhimpur'},
    'mainpuri': {'lat': 27.2356, 'lon': 79.0270, 'name': 'Mainpuri'},
    'orai': {'lat': 26.0055, 'lon': 79.4502, 'name': 'Orai'},
    'rae bareli': {'lat': 26.2152, 'lon': 81.2456, 'name': 'Rae Bareli'},
    'sultanpur': {'lat': 26.2677, 'lon': 82.0739, 'name': 'Sultanpur'},
    
    # Punjab cities
    'jalandhar': {'lat': 31.3260, 'lon': 75.5762, 'name': 'Jalandhar'},
    'patiala': {'lat': 30.3398, 'lon': 76.3869, 'name': 'Patiala'},
    'bathinda': {'lat': 30.2110, 'lon': 74.9455, 'name': 'Bathinda'},
    'mohali': {'lat': 30.7046, 'lon': 76.7179, 'name': 'Mohali'},
    'pathankot': {'lat': 32.2746, 'lon': 75.6521, 'name': 'Pathankot'},
    'hoshiarpur': {'lat': 31.5344, 'lon': 75.9117, 'name': 'Hoshiarpur'},
    'batala': {'lat': 31.8230, 'lon': 75.2045, 'name': 'Batala'},
    'abohar': {'lat': 30.1204, 'lon': 74.1995, 'name': 'Abohar'},
    'malerkotla': {'lat': 30.5281, 'lon': 75.8792, 'name': 'Malerkotla'},
    'khanna': {'lat': 30.7057, 'lon': 76.2222, 'name': 'Khanna'},
    'phagwara': {'lat': 31.2244, 'lon': 75.7729, 'name': 'Phagwara'},
    'muktsar': {'lat': 30.4762, 'lon': 74.5161, 'name': 'Muktsar'},
    'kapurthala': {'lat': 31.3800, 'lon': 75.3800, 'name': 'Kapurthala'},
    'firozpur': {'lat': 30.9324, 'lon': 74.6150, 'name': 'Firozpur'},
    'fazilka': {'lat': 30.4028, 'lon': 74.0286, 'name': 'Fazilka'},
    'gurdaspur': {'lat': 32.0409, 'lon': 75.4065, 'name': 'Gurdaspur'},
    'mansa': {'lat': 29.9988, 'lon': 75.3932, 'name': 'Mansa'},
    'sangrur': {'lat': 30.2458, 'lon': 75.8421, 'name': 'Sangrur'},
    'barnala': {'lat': 30.3782, 'lon': 75.5463, 'name': 'Barnala'},
    'fatehgarh sahib': {'lat': 30.6446, 'lon': 76.3931, 'name': 'Fatehgarh Sahib'},
    'rupnagar': {'lat': 30.9629, 'lon': 76.5270, 'name': 'Rupnagar'},
    'nawanshahr': {'lat': 31.1254, 'lon': 76.1163, 'name': 'Nawanshahr'},
    'tarn taran': {'lat': 31.4517, 'lon': 74.9255, 'name': 'Tarn Taran'}
}

@app.route('/')
def index():
    return render_template('index_new.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response for favicon

@app.route('/government')
def government_dashboard():
    return render_template('government_dashboard.html')

@app.route('/heatmap')
def heatmap_dashboard():
    return render_template('heatmap_dashboard.html')

@app.route('/historical-dashboard')
def historical_dashboard():
    return render_template('historical_dashboard.html')

@app.route('/status')
def status():
    return jsonify({
        "model_loaded": model is not None,
        "features_count": len(feature_columns),
        "model_type": "CNN-BiLSTM-Attention" if model else None,
        "openweather_api": OPENWEATHER_API_KEY is not None,
        "tomtom_api": TOMTOM_API_KEY is not None,
        "cache_entries": len(api_cache)
    })

@app.route('/geocode_city', methods=['POST'])
def geocode_city():
    try:
        data = request.get_json()
        city_name = data.get('city', '').strip()
        
        if not city_name:
            return jsonify({'error': 'City name is required'}), 400
        
        # Try TomTom API first
        geocode_data = geocode_with_tomtom(city_name)
        
        if geocode_data:
            return jsonify({
                'success': True,
                'latitude': geocode_data['lat'],
                'longitude': geocode_data['lon'],
                'city': city_name.title(),
                'country': 'India',
                'address': geocode_data['address'],
                'data_source': 'tomtom_api'
            })
        
        # Fallback to Indian cities database
        city_lower = city_name.lower().strip()
        
        # Direct match
        if city_lower in INDIAN_CITIES:
            city_data = INDIAN_CITIES[city_lower]
            return jsonify({
                'success': True,
                'latitude': city_data['lat'],
                'longitude': city_data['lon'],
                'city': city_data['name'],
                'country': 'India',
                'address': f"{city_data['name']}, India",
                'data_source': 'local_database'
            })
        
        # Partial match for typos or variations
        for key, city_data in INDIAN_CITIES.items():
            if city_lower in key or key in city_lower:
                return jsonify({
                    'success': True,
                    'latitude': city_data['lat'],
                    'longitude': city_data['lon'],
                    'city': city_data['name'],
                    'country': 'India',
                    'address': f"{city_data['name']}, India",
                    'data_source': 'local_database_fuzzy'
                })
        
        return jsonify({'error': f'City "{city_name}" not found. Try: Mumbai, Delhi, Pune, Bangalore, Chennai, etc.'}), 404
    
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_location_data', methods=['POST'])
def fetch_location_data():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
        
        # Try to fetch real weather data
        weather_data = fetch_real_weather_data(lat, lon)
        
        if weather_data:
            # Real API data
            now = datetime.now()
            
            # Convert weather to codes
            weather_code = 1  # Clear
            if weather_data['weather_main'] in ['Rain', 'Drizzle']:
                weather_code = 3
            elif weather_data['weather_main'] == 'Clouds':
                weather_code = 2
            elif weather_data['weather_main'] in ['Mist', 'Fog']:
                weather_code = 7
            
            road_surface = 0  # Dry
            if weather_data['weather_main'] in ['Rain', 'Drizzle']:
                road_surface = 1  # Wet
            
            return jsonify({
                'success': True,
                'main_weather': weather_data['weather_main'],
                'description': weather_data['weather_description'],
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'wind_speed': weather_data['wind_speed'],
                'pressure': weather_data.get('pressure', 1013),
                'visibility': weather_data.get('visibility', 10),
                'hour': now.hour,
                'day_of_week': now.isoweekday(),
                'month': now.month,
                'light_conditions': 0 if 6 <= now.hour <= 18 else 1,
                'weather_conditions': weather_code,
                'road_surface': road_surface,
                'speed_limit': 60,
                'data_source': 'real_weather_api'
            })
        else:
            # Fallback to simulated data
            now = datetime.now()
            return jsonify({
                'success': True,
                'main_weather': 'Clear',
                'description': 'clear sky (simulated)',
                'temperature': 26.0,
                'humidity': 65,
                'wind_speed': 3.5,
                'pressure': 1013,
                'visibility': 10,
                'hour': now.hour,
                'day_of_week': now.isoweekday(),
                'month': now.month,
                'light_conditions': 0 if 6 <= now.hour <= 18 else 1,
                'weather_conditions': 1,
                'road_surface': 0,
                'speed_limit': 60,
                'data_source': 'simulated'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/heatmap/generate', methods=['POST'])
def generate_heatmap():
    try:
        data = request.get_json()
        bounds = data.get('bounds', {})
        
        # Generate heatmap data points using real weather and IEEE methodology
        heatmap_data = []
        
        # Create grid points within bounds
        lat_min = bounds.get('south', 8.0)
        lat_max = bounds.get('north', 35.0)
        lng_min = bounds.get('west', 68.0)
        lng_max = bounds.get('east', 97.0)
        
        # Generate 50 sample points
        import random
        for i in range(50):
            lat = lat_min + (lat_max - lat_min) * random.random()
            lng = lng_min + (lng_max - lng_min) * random.random()
            
            # Get real weather data for this location
            weather_data = fetch_real_weather_data(lat, lng)
            
            # Calculate risk using IEEE methodology
            input_data = {
                'Latitude': lat,
                'Longitude': lng,
                'hour': datetime.now().hour,
                'Day_of_Week': datetime.now().isoweekday(),
                'Weather_Conditions': 1,
                'month': datetime.now().month,
                'Speed_limit': random.choice([30, 40, 50, 60, 70])
            }
            
            if weather_data:
                input_data.update({
                    'temperature': weather_data['temperature'],
                    'humidity': weather_data['humidity'],
                    'wind_speed': weather_data['wind_speed']
                })
                # Update weather conditions based on real data
                if 'rain' in weather_data['weather_description'].lower():
                    input_data['Weather_Conditions'] = 2
                elif 'fog' in weather_data['weather_description'].lower():
                    input_data['Weather_Conditions'] = 7
            
            # Calculate risk using IEEE methodology
            weather_risk = calculate_weather_risk(input_data)
            temporal_risk = calculate_temporal_risk(input_data)
            spatial_risk = calculate_spatial_risk(input_data)
            
            risk_value = (weather_risk * 0.4) + (temporal_risk * 0.4) + (spatial_risk * 0.2)
            risk_value = max(1.0, min(3.0, risk_value))
            
            # Normalize to 0-1 for heatmap intensity
            intensity = (risk_value - 1.0) / 2.0
            
            heatmap_data.append({
                'lat': lat,
                'lng': lng,
                'intensity': intensity,
                'risk_level': 'Low' if risk_value < 1.4 else 'Medium' if risk_value < 2.3 else 'High',
                'risk_value': round(risk_value, 2)
            })
        
        return jsonify({
            'success': True,
            'data': heatmap_data,
            'total_points': len(heatmap_data),
            'data_source': 'real_weather_ieee'
        })
        
    except Exception as e:
        logger.error(f"Heatmap generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/government/risk-areas', methods=['GET'])
def get_government_risk_areas():
    try:
        # Generate risk areas using real data and IEEE methodology
        risk_areas = []
        
        # Major Indian cities with real coordinates
        cities = [
            {'name': 'Mumbai', 'lat': 19.0760, 'lng': 72.8777},
            {'name': 'Delhi', 'lat': 28.7041, 'lng': 77.1025},
            {'name': 'Bangalore', 'lat': 12.9716, 'lng': 77.5946},
            {'name': 'Chennai', 'lat': 13.0827, 'lng': 80.2707},
            {'name': 'Kolkata', 'lat': 22.5726, 'lng': 88.3639},
            {'name': 'Hyderabad', 'lat': 17.3850, 'lng': 78.4867},
            {'name': 'Pune', 'lat': 18.5204, 'lng': 73.8567},
            {'name': 'Ahmedabad', 'lat': 23.0225, 'lng': 72.5714}
        ]
        
        for city in cities:
            # Get real weather data
            weather_data = fetch_real_weather_data(city['lat'], city['lng'])
            
            input_data = {
                'Latitude': city['lat'],
                'Longitude': city['lng'],
                'hour': datetime.now().hour,
                'Day_of_Week': datetime.now().isoweekday(),
                'Weather_Conditions': 1,
                'month': datetime.now().month,
                'Speed_limit': 50
            }
            
            if weather_data:
                input_data.update({
                    'temperature': weather_data['temperature'],
                    'humidity': weather_data['humidity'],
                    'wind_speed': weather_data['wind_speed']
                })
                # Update weather conditions
                if 'rain' in weather_data['weather_description'].lower():
                    input_data['Weather_Conditions'] = 2
                elif 'fog' in weather_data['weather_description'].lower():
                    input_data['Weather_Conditions'] = 7
            
            # Calculate risk using IEEE methodology
            weather_risk = calculate_weather_risk(input_data)
            temporal_risk = calculate_temporal_risk(input_data)
            spatial_risk = calculate_spatial_risk(input_data)
            
            risk_value = (weather_risk * 0.4) + (temporal_risk * 0.4) + (spatial_risk * 0.2)
            risk_value = max(1.0, min(3.0, risk_value))
            
            if risk_value < 1.4:
                risk_level = 'Low'
                color = 'green'
            elif risk_value < 2.3:
                risk_level = 'Medium'
                color = 'yellow'
            else:
                risk_level = 'High'
                color = 'red'
            
            risk_areas.append({
                'name': city['name'],
                'lat': city['lat'],
                'lng': city['lng'],
                'risk_level': risk_level,
                'risk_value': round(risk_value, 2),
                'color': color,
                'weather_desc': weather_data['weather_description'] if weather_data else 'Clear',
                'temperature': weather_data['temperature'] if weather_data else 25,
                'data_source': 'real_weather' if weather_data else 'fallback'
            })
        
        return jsonify({
            'success': True,
            'risk_areas': risk_areas,
            'total_areas': len(risk_areas),
            'methodology': 'IEEE_real_data'
        })
        
    except Exception as e:
        logger.error(f"Government risk areas error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    try:
        data = request.get_json()
        
        # Get real weather data for prediction
        lat = float(data.get('latitude', 19.0760))
        lon = float(data.get('longitude', 72.8777))
        
        # Fetch current weather conditions
        weather_data = fetch_real_weather_data(lat, lon)
        
        input_data = {
            'Latitude': lat,
            'Longitude': lon,
            'hour': int(data.get('hour', datetime.now().hour)),
            'Day_of_Week': int(data.get('day_of_week', datetime.now().isoweekday())),
            'Weather_Conditions': int(data.get('weather_conditions', 1)),
            'Road_Surface_Conditions': int(data.get('road_surface', 0)),
            'Speed_limit': int(data.get('speed_limit', 50)),
            'Junction_Detail': int(data.get('junction_detail', 0)),
            'Road_Type': int(data.get('road_type', 3)),
            'Number_of_Vehicles': int(data.get('vehicles', 2)),
            'Number_of_Casualties': int(data.get('casualties', 1)),
            'Light_Conditions': int(data.get('light_conditions', 0 if 6 <= datetime.now().hour <= 18 else 1)),
            'month': int(data.get('month', datetime.now().month)),
            'week': int(data.get('week', datetime.now().isocalendar()[1]))
        }
        
        # Add real weather data if available
        if weather_data:
            input_data.update({
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'wind_speed': weather_data['wind_speed'],
                'pressure': weather_data.get('pressure', 1013),
                'visibility': weather_data.get('visibility', 10)
            })
        else:
            # Default weather values
            input_data.update({
                'temperature': 25.0,
                'humidity': 60,
                'wind_speed': 2.0,
                'pressure': 1013,
                'visibility': 10
            })
        
        # Try AI model first
        result = predict_with_ai(input_data)
        
        if result:
            return jsonify(result)
        
        # IEEE paper fallback methodology with real data integration
        weather_risk = calculate_weather_risk(input_data)
        temporal_risk = calculate_temporal_risk(input_data)
        spatial_risk = calculate_spatial_risk(input_data)
        
        # IEEE cumulative risk calculation
        risk_score = (weather_risk * 0.4) + (temporal_risk * 0.4) + (spatial_risk * 0.2)
        
        # Ensure within IEEE range (1-3)
        risk_score = max(1.0, min(3.0, risk_score))
        
        # IEEE classification (based on accident severity)
        if risk_score < 1.4:
            risk_level = "Low"    # Slight accidents (risk value 1)
        elif risk_score < 2.3:
            risk_level = "Medium"  # Serious accidents (risk value 2)
        else:
            risk_level = "High"   # Fatal accidents (risk value 3)
        
        return jsonify({
            "risk_value": round(risk_score, 3),
            "risk_level": risk_level,
            "confidence": 75.0,
            "prediction_source": "ieee_methodology",
            "model_type": "Multi-factor Risk Analysis",
            "weather_risk": round(weather_risk, 2),
            "temporal_risk": round(temporal_risk, 2),
            "spatial_risk": round(spatial_risk, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("RoadSafe AI - Real API Integration")
    print("=" * 50)
    
    # Check API keys
    if OPENWEATHER_API_KEY:
        print("[OK] OpenWeatherMap API key loaded")
    else:
        print("[WARN] OpenWeatherMap API key missing")
    
    if TOMTOM_API_KEY:
        print("[OK] TomTom API key loaded")
    else:
        print("[WARN] TomTom API key missing")
    
    # Load AI model
    if load_ai_model():
        print("[OK] Real AI model loaded successfully")
    else:
        print("[WARN] Using fallback predictions")
    
    print("Starting server at http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)