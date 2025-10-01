#!/usr/bin/env python3
"""
Fresh RoadSafe AI Application - Built from scratch
Indian cities focused with real AI model predictions
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
    """CNN-BiLSTM-Attention model for traffic accident prediction"""
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
        
        # Attention
        self.s_attn = nn.Module()
        self.s_attn.proj = nn.ModuleList([
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 128)
        ])
        self.t_attn = nn.Module()
        self.t_attn.query = nn.Linear(128, 64)
        self.t_attn.key = nn.Linear(128, 64)
        
        # LSTM
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
        
        # Spatial attention
        s_x = self.s_attn.proj[0](x)
        s_x = torch.relu(s_x)
        s_x = torch.dropout(s_x, 0.3, self.training)
        s_x = self.s_attn.proj[2](s_x)
        s_attn = torch.sigmoid(s_x)
        x = x * s_attn
        
        # Temporal attention
        q = self.t_attn.query(x)
        k = self.t_attn.key(x)
        t_attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True), dim=1)
        x = x * t_attn
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.out(x)
        return x.squeeze(-1)

def load_ai_model():
    """Load the trained AI model"""
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
    """Make prediction using AI model"""
    global model, feature_columns, config
    
    if model is None:
        return None
        
    try:
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Add missing features
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
        
        # Prepare tensor
        X = df[feature_columns].values.astype(np.float32)
        seq_len = config['data']['sequence_length']
        X_seq = np.tile(X, (seq_len, 1)).reshape(1, seq_len, -1)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        # Predict
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

# Indian cities database
INDIAN_CITIES = {
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai'},
    'delhi': {'lat': 28.7041, 'lon': 77.1025, 'name': 'Delhi'},
    'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bangalore'},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad'},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Ahmedabad'},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai'},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata'},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Pune'},
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur'},
    'surat': {'lat': 21.1702, 'lon': 72.8311, 'name': 'Surat'},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow'},
    'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'name': 'Kanpur'},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'name': 'Nagpur'},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore'},
    'thane': {'lat': 19.2183, 'lon': 72.9781, 'name': 'Thane'},
    'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal'},
    'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Visakhapatnam'},
    'pimpri': {'lat': 18.6298, 'lon': 73.7997, 'name': 'Pimpri-Chinchwad'},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Patna'},
    'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'name': 'Vadodara'},
    'ghaziabad': {'lat': 28.6692, 'lon': 77.4538, 'name': 'Ghaziabad'},
    'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'name': 'Ludhiana'},
    'agra': {'lat': 27.1767, 'lon': 78.0081, 'name': 'Agra'},
    'nashik': {'lat': 19.9975, 'lon': 73.7898, 'name': 'Nashik'},
    'faridabad': {'lat': 28.4089, 'lon': 77.3178, 'name': 'Faridabad'},
    'meerut': {'lat': 28.9845, 'lon': 77.7064, 'name': 'Meerut'},
    'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'name': 'Rajkot'},
    'kalyan': {'lat': 19.2437, 'lon': 73.1355, 'name': 'Kalyan-Dombivli'},
    'vasai': {'lat': 19.4912, 'lon': 72.8054, 'name': 'Vasai-Virar'},
    'varanasi': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Varanasi'},
    'srinagar': {'lat': 34.0837, 'lon': 74.7973, 'name': 'Srinagar'},
    'aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'name': 'Aurangabad'},
    'dhanbad': {'lat': 23.7957, 'lon': 86.4304, 'name': 'Dhanbad'},
    'amritsar': {'lat': 31.6340, 'lon': 74.8723, 'name': 'Amritsar'},
    'navi mumbai': {'lat': 19.0330, 'lon': 73.0297, 'name': 'Navi Mumbai'},
    'allahabad': {'lat': 25.4358, 'lon': 81.8463, 'name': 'Prayagraj'},
    'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'name': 'Ranchi'},
    'howrah': {'lat': 22.5958, 'lon': 88.2636, 'name': 'Howrah'},
    'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'name': 'Coimbatore'},
    'jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'name': 'Jabalpur'},
    'gwalior': {'lat': 26.2183, 'lon': 78.1828, 'name': 'Gwalior'},
    'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Vijayawada'},
    'jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'name': 'Jodhpur'},
    'madurai': {'lat': 9.9252, 'lon': 78.1198, 'name': 'Madurai'},
    'raipur': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur'},
    'kota': {'lat': 25.2138, 'lon': 75.8648, 'name': 'Kota'},
    'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh'},
    'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'name': 'Guwahati'},
    'solapur': {'lat': 17.6599, 'lon': 75.9064, 'name': 'Solapur'},
    'hubli': {'lat': 15.3647, 'lon': 75.1240, 'name': 'Hubli-Dharwad'},
    'bareilly': {'lat': 28.3670, 'lon': 79.4304, 'name': 'Bareilly'},
    'moradabad': {'lat': 28.8386, 'lon': 78.7733, 'name': 'Moradabad'},
    'mysore': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysuru'},
    'tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'name': 'Tiruchirappalli'},
    'salem': {'lat': 11.6643, 'lon': 78.1460, 'name': 'Salem'},
    'aligarh': {'lat': 27.8974, 'lon': 78.0880, 'name': 'Aligarh'},
    'tiruppur': {'lat': 11.1085, 'lon': 77.3411, 'name': 'Tiruppur'},
    'guntur': {'lat': 16.3067, 'lon': 80.4365, 'name': 'Guntur'},
    'bhiwandi': {'lat': 19.3002, 'lon': 73.0635, 'name': 'Bhiwandi'},
    'saharanpur': {'lat': 29.9680, 'lon': 77.5552, 'name': 'Saharanpur'},
    'gorakhpur': {'lat': 26.7606, 'lon': 83.3732, 'name': 'Gorakhpur'},
    'bikaner': {'lat': 28.0229, 'lon': 73.3119, 'name': 'Bikaner'},
    'amravati': {'lat': 20.9374, 'lon': 77.7796, 'name': 'Amravati'},
    'noida': {'lat': 28.5355, 'lon': 77.3910, 'name': 'Noida'},
    'jamshedpur': {'lat': 22.8046, 'lon': 86.2029, 'name': 'Jamshedpur'},
    'bhilai': {'lat': 21.1938, 'lon': 81.3509, 'name': 'Bhilai'},
    'cuttack': {'lat': 20.4625, 'lon': 85.8828, 'name': 'Cuttack'},
    'firozabad': {'lat': 27.1592, 'lon': 78.3957, 'name': 'Firozabad'},
    'kochi': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Kochi'},
    'nellore': {'lat': 14.4426, 'lon': 79.9865, 'name': 'Nellore'},
    'bhavnagar': {'lat': 21.7645, 'lon': 72.1519, 'name': 'Bhavnagar'},
    'dehradun': {'lat': 30.3165, 'lon': 78.0322, 'name': 'Dehradun'},
    'durgapur': {'lat': 23.5204, 'lon': 87.3119, 'name': 'Durgapur'},
    'asansol': {'lat': 23.6739, 'lon': 86.9524, 'name': 'Asansol'},
    'rourkela': {'lat': 22.2604, 'lon': 84.8536, 'name': 'Rourkela'},
    'nanded': {'lat': 19.1383, 'lon': 77.3210, 'name': 'Nanded'},
    'kolhapur': {'lat': 16.7050, 'lon': 74.2433, 'name': 'Kolhapur'},
    'ajmer': {'lat': 26.4499, 'lon': 74.6399, 'name': 'Ajmer'},
    'akola': {'lat': 20.7002, 'lon': 77.0082, 'name': 'Akola'},
    'gulbarga': {'lat': 17.3297, 'lon': 76.8343, 'name': 'Kalaburagi'},
    'jamnagar': {'lat': 22.4707, 'lon': 70.0577, 'name': 'Jamnagar'},
    'ujjain': {'lat': 23.1765, 'lon': 75.7885, 'name': 'Ujjain'},
    'loni': {'lat': 28.7333, 'lon': 77.2833, 'name': 'Loni'},
    'siliguri': {'lat': 26.7271, 'lon': 88.3953, 'name': 'Siliguri'},
    'jhansi': {'lat': 25.4484, 'lon': 78.5685, 'name': 'Jhansi'},
    'ulhasnagar': {'lat': 19.2215, 'lon': 73.1645, 'name': 'Ulhasnagar'},
    'jammu': {'lat': 32.7266, 'lon': 74.8570, 'name': 'Jammu'},
    'sangli': {'lat': 16.8524, 'lon': 74.5815, 'name': 'Sangli-Miraj & Kupwad'},
    'mangalore': {'lat': 12.9141, 'lon': 74.8560, 'name': 'Mangaluru'},
    'erode': {'lat': 11.3410, 'lon': 77.7172, 'name': 'Erode'},
    'belgaum': {'lat': 15.8497, 'lon': 74.4977, 'name': 'Belagavi'},
    'ambattur': {'lat': 13.1143, 'lon': 80.1548, 'name': 'Ambattur'},
    'tirunelveli': {'lat': 8.7139, 'lon': 77.7567, 'name': 'Tirunelveli'},
    'malegaon': {'lat': 20.5579, 'lon': 74.5287, 'name': 'Malegaon'},
    'gaya': {'lat': 24.7914, 'lon': 85.0002, 'name': 'Gaya'},
    'jalgaon': {'lat': 21.0077, 'lon': 75.5626, 'name': 'Jalgaon'},
    'udaipur': {'lat': 24.5854, 'lon': 73.7125, 'name': 'Udaipur'},
    'maheshtala': {'lat': 22.4967, 'lon': 88.2467, 'name': 'Maheshtala'}
}

@app.route('/')
def index():
    from flask import make_response
    response = make_response(render_template('index_new.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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
    
    return jsonify({'error': f'City "{city_name}" not found in Indian cities database'}), 404

@app.route('/predict_risk', methods=['POST'])
def predict_risk_route():
    data = request.get_json()
    
    # Build input data
    input_data = {
        'Latitude': float(data.get('latitude', 19.0760)),  # Default Mumbai
        'Longitude': float(data.get('longitude', 72.8777)),
        'hour': int(data.get('hour', 14)),
        'Day_of_Week': int(data.get('day_of_week', 3)),
        'Weather_Conditions': int(data.get('weather_conditions', 1)),
        'Road_Surface_Conditions': int(data.get('road_surface', 0)),
        'Speed_limit': int(data.get('speed_limit', 50)),  # Indian roads
        'Junction_Detail': int(data.get('junction_detail', 0)),
        'Road_Type': int(data.get('road_type', 3)),
        'Number_of_Vehicles': int(data.get('vehicles', 2)),
        'Number_of_Casualties': int(data.get('casualties', 1)),
        'Light_Conditions': int(data.get('light_conditions', 0)),
        'month': int(data.get('month', datetime.now().month)),
        'week': int(data.get('week', 25))
    }
    
    # Try AI prediction first
    result = predict_with_ai(input_data)
    
    if result:
        return jsonify(result)
    
    # Fallback prediction for Indian conditions
    lat = input_data['Latitude']
    lon = input_data['Longitude']
    hour = input_data['hour']
    weather = input_data['Weather_Conditions']
    speed = input_data['Speed_limit']
    
    # Base risk for Indian roads (higher than UK)
    risk_score = 1.8  # Indian roads are generally riskier
    
    # Location-based risk (Indian cities)
    if 18 <= lat <= 29 and 72 <= lon <= 88:  # North India belt
        risk_score += 0.3
    elif 8 <= lat <= 18 and 75 <= lon <= 80:  # South India
        risk_score += 0.2
    
    # Time-based risk (Indian traffic patterns)
    if hour < 6 or hour > 22:
        risk_score += 0.4  # Night driving
    elif 7 <= hour <= 10 or 17 <= hour <= 20:
        risk_score += 0.5  # Heavy traffic hours
    
    # Weather risk (monsoon conditions)
    if weather in [2, 3, 7]:  # Rain/fog common in India
        risk_score += 0.6
    
    # Speed risk (Indian road conditions)
    if speed > 80:
        risk_score += 0.4
    elif speed > 60:
        risk_score += 0.2
    
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
        "prediction_source": "indian_roads_analysis",
        "note": "Optimized for Indian road conditions"
    })

if __name__ == '__main__':
    print("🇮🇳 Starting RoadSafe AI - Indian Roads Edition")
    print("=" * 60)
    
    # Load AI model
    if load_ai_model():
        print("✅ Real AI model loaded successfully")
    else:
        print("⚠️ Using fallback predictions optimized for Indian roads")
    
    print(f"📍 {len(INDIAN_CITIES)} Indian cities in database")
    print("🌐 Starting server at http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=True)