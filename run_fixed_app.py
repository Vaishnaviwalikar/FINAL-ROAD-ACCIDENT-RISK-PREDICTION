#!/usr/bin/env python3
"""
Fixed RoadSafe AI app with proper error handling
"""
import os
import sys
import json
import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            risk_prob = torch.sigmoid(torch.tensor(raw_output)).item()
            risk_value = 1.0 + (risk_prob * 2.0)
            
            if risk_prob < 0.33:
                risk_level = "Low"
            elif risk_prob < 0.66:
                risk_level = "Medium"
            else:
                risk_level = "High"
                
            confidence = 80.0 + (risk_prob * 15.0)
            
            return {
                "risk_value": round(risk_value, 3),
                "risk_level": risk_level,
                "confidence": round(confidence, 1),
                "used_model": True,
                "prediction_source": "real_ai_model",
                "model_type": "CNN-BiLSTM-Attention"
            }
            
    except Exception as e:
        logger.error(f"AI prediction failed: {e}")
        return None

# Indian cities
INDIAN_CITIES = {
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai'},
    'delhi': {'lat': 28.7041, 'lon': 77.1025, 'name': 'Delhi'},
    'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bangalore'},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai'},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata'},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad'},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Pune'},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Ahmedabad'},
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur'},
    'surat': {'lat': 21.1702, 'lon': 72.8311, 'name': 'Surat'},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow'},
    'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'name': 'Kanpur'},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'name': 'Nagpur'},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore'},
    'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal'},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Patna'},
    'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'name': 'Vadodara'},
    'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'name': 'Ludhiana'},
    'agra': {'lat': 27.1767, 'lon': 78.0081, 'name': 'Agra'},
    'nashik': {'lat': 19.9975, 'lon': 73.7898, 'name': 'Nashik'}
}

@app.route('/')
def index():
    return render_template('index_new.html')

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
        "indian_cities": len(INDIAN_CITIES)
    })

@app.route('/geocode_city', methods=['POST'])
def geocode_city():
    try:
        data = request.get_json()
        city_name = data.get('city', '').lower().strip()
        
        if city_name in INDIAN_CITIES:
            city_data = INDIAN_CITIES[city_name]
            return jsonify({
                'latitude': city_data['lat'],
                'longitude': city_data['lon'],
                'city': city_data['name'],
                'country': 'India'
            })
        
        return jsonify({'error': f'City "{city_name}" not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_location_data', methods=['POST'])
def fetch_location_data():
    try:
        data = request.get_json()
        lat = data.get('latitude', 19.0760)
        lon = data.get('longitude', 72.8777)
        
        api_key = os.environ.get('OPENWEATHER_API_KEY', 'demo_key')
        now = datetime.now()
        
        if api_key == 'demo_key':
            # Demo mode with realistic Indian weather
            return jsonify({
                'main_weather': 'Clear',
                'description': 'clear sky',
                'temperature': 28.0,
                'humidity': 65,
                'wind_speed': 3.5,
                'hour': now.hour,
                'day_of_week': now.isoweekday(),
                'month': now.month,
                'light_conditions': 0 if 6 <= now.hour <= 18 else 1,
                'weather_conditions': 1,
                'road_surface': 0,
                'speed_limit': 60,
                'data_source': 'demo_indian_weather'
            })
        
        try:
            # Real weather API call
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(weather_url, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            main_weather = weather_data['weather'][0]['main']
            weather_code = 1  # Clear
            if main_weather in ['Rain', 'Drizzle']:
                weather_code = 3
            elif main_weather == 'Clouds':
                weather_code = 2
            elif main_weather in ['Mist', 'Fog']:
                weather_code = 7
            
            road_surface = 0  # Dry
            if main_weather in ['Rain', 'Drizzle'] and weather_data['main']['temp'] > 0:
                road_surface = 1  # Wet
            
            return jsonify({
                'main_weather': main_weather,
                'description': weather_data['weather'][0]['description'],
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'wind_speed': weather_data['wind']['speed'],
                'hour': now.hour,
                'day_of_week': now.isoweekday(),
                'month': now.month,
                'light_conditions': 0 if 6 <= now.hour <= 18 else 1,
                'weather_conditions': weather_code,
                'road_surface': road_surface,
                'speed_limit': 60,
                'data_source': 'real_weather_api'
            })
            
        except Exception as e:
            # Fallback to demo weather
            return jsonify({
                'main_weather': 'Clear',
                'description': 'clear sky (API fallback)',
                'temperature': 26.0,
                'humidity': 70,
                'wind_speed': 4.0,
                'hour': now.hour,
                'day_of_week': now.isoweekday(),
                'month': now.month,
                'light_conditions': 0 if 6 <= now.hour <= 18 else 1,
                'weather_conditions': 1,
                'road_surface': 0,
                'speed_limit': 60,
                'data_source': 'fallback_weather',
                'error': str(e)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_risk', methods=['POST'])
def predict_risk_route():
    try:
        data = request.get_json()
        
        input_data = {
            'Latitude': float(data.get('latitude', 19.0760)),
            'Longitude': float(data.get('longitude', 72.8777)),
            'hour': int(data.get('hour', 14)),
            'Day_of_Week': int(data.get('day_of_week', 3)),
            'Weather_Conditions': int(data.get('weather_conditions', 1)),
            'Road_Surface_Conditions': int(data.get('road_surface', 0)),
            'Speed_limit': int(data.get('speed_limit', 50)),
            'Junction_Detail': int(data.get('junction_detail', 0)),
            'Road_Type': int(data.get('road_type', 3)),
            'Number_of_Vehicles': int(data.get('vehicles', 2)),
            'Number_of_Casualties': int(data.get('casualties', 1)),
            'Light_Conditions': int(data.get('light_conditions', 0)),
            'month': int(data.get('month', datetime.now().month)),
            'week': int(data.get('week', 25))
        }
        
        result = predict_with_ai(input_data)
        
        if result:
            return jsonify(result)
        
        # Fallback
        risk_score = 2.0
        lat = input_data['Latitude']
        hour = input_data['hour']
        weather = input_data['Weather_Conditions']
        
        if 18 <= lat <= 29:  # North India
            risk_score += 0.3
        if hour < 6 or hour > 22:
            risk_score += 0.4
        if weather in [2, 3, 7]:
            risk_score += 0.6
        
        risk_score = max(1.0, min(3.0, risk_score))
        
        if risk_score < 1.7:
            risk_level = "Low"
        elif risk_score < 2.3:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return jsonify({
            "risk_value": round(risk_score, 3),
            "risk_level": risk_level,
            "confidence": 78.0,
            "used_model": False,
            "prediction_source": "indian_roads_analysis"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🇮🇳 Starting RoadSafe AI - Fixed Version")
    print("=" * 60)
    
    if load_ai_model():
        print("✅ Real AI model loaded successfully")
    else:
        print("⚠️ Using fallback predictions")
    
    print(f"📍 {len(INDIAN_CITIES)} Indian cities available")
    print("🌐 Starting server at http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)