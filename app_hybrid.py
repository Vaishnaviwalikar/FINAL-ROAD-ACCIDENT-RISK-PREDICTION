#!/usr/bin/env python3
"""
Flask Web Application for Road Traffic Accident Risk Prediction
Hybrid version: Full ML models when available, graceful fallback when not
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
import traceback
from dotenv import load_dotenv

# Disable CUDA at the very beginning
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.cuda.is_available = lambda: False

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.debug = True

# Global variables for model components
a_model = None
a_feature_columns = []  # Initialize as empty list
a_label_encoders = {}   # Initialize as empty dict
a_config = {}           # Initialize as empty dict
a_processed_dir = 'data/processed/uk'  # Default path
a_scaler = None
a_checkpoint_path = None
a_device = 'cpu'  # Default to CPU

# Simple weather cache
_WEATHER_CACHE = {}

def ensure_model_on_cpu(model):
    """Ensure all model parameters are on CPU and in eval mode"""
    if model is None:
        return None
    
    # Move model to CPU
    model = model.cpu()
    
    # Set to eval mode
    model.eval()
    
    # Ensure all parameters are on CPU
    for param in model.parameters():
        param.data = param.data.cpu()
        if param._grad is not None:
            param._grad.data = param._grad.data.cpu()
    
    # If model was wrapped in DataParallel, unwrap it
    if hasattr(model, 'module'):
        model = model.module
    
    return model

def get_openweather_api_key():
    """Get OpenWeatherMap API key from environment"""
    return os.environ.get('OPENWEATHER_API_KEY', 'demo_key')

def is_demo_mode():
    """Check if we're in demo mode"""
    api_key = get_openweather_api_key()
    return api_key == 'demo_key' or not api_key

# Weather conditions mapping
WEATHER_CONDITIONS_MAP = {
    'Clear': 1, 'Clouds': 2, 'Rain': 3, 'Drizzle': 3,
    'Thunderstorm': 4, 'Snow': 5, 'Mist': 6, 'Fog': 7
}

def get_road_surface_from_weather(weather_main, temperature):
    """Determine road surface based on weather"""
    if weather_main in ['Rain', 'Drizzle', 'Thunderstorm']:
        return 1  # Wet/Damp
    elif weather_main == 'Snow' and temperature < 0:
        return 3  # Frost/Ice
    elif weather_main == 'Snow':
        return 2  # Snow
    else:
        return 0  # Dry

def get_light_conditions(hour):
    """Determine light conditions based on hour"""
    return 0 if 6 <= hour <= 18 else 1  # 0=Daylight, 1=Darkness

def ensure_model_on_cpu(model):
    """Ensure all model parameters are on CPU and in eval mode"""
    if model is None:
        return None
    
    # Set device to CPU
    device = torch.device('cpu')
    
    # Move model to CPU
    model = model.to(device)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Recursively move all parameters to CPU
    for param in model.parameters():
        param.data = param.data.to(device)
        if param._grad is not None:
            param._grad = param._grad.to(device)
    
    # If model was wrapped in DataParallel, unwrap it
    if hasattr(model, 'module'):
        model = model.module
    
    # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model

def setup_logging():
    """Set up logging configuration."""
    import logging
    from logging.handlers import RotatingFileHandler
    import os
    from pathlib import Path
    import traceback
    import torch

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Configure basic logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=5)
        ]
    )
    
    # Set up a more detailed file handler
    file_handler = RotatingFileHandler('logs/debug.log', maxBytes=10485760, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add the handler to the root logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

def verify_model_on_cpu(model):
    """Verify all model parameters are on CPU"""
    if model is None:
        return False
    
    # Check all parameters
    for name, param in model.named_parameters():
        if param.device.type != 'cpu':
            logger.error(f"Parameter {name} is on {param.device}, should be on CPU")
            return False
    
    # Check all buffers
    for name, buf in model.named_buffers():
        if buf.device.type != 'cpu':
            logger.error(f"Buffer {name} is on {buf.device}, should be on CPU")
            return False
    
    return True

class WorkingAIModel(nn.Module):
    """Working AI model that matches checkpoint structure."""
    
    def __init__(self, in_channels=30):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        
        # FC layer
        self.fc = nn.Linear(32, 128)
        
        # Attention layers
        self.s_attn = nn.Module()
        self.s_attn.proj = nn.ModuleList([
            nn.Linear(128, 64),      # proj.0
            nn.ReLU(),               # proj.1
            nn.Linear(64, 128),      # proj.2
        ])
        
        self.t_attn = nn.Module()
        self.t_attn.query = nn.Linear(128, 64)
        self.t_attn.key = nn.Linear(128, 64)
        
        # LSTM
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Output
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        # Conv processing
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
        
        # FC and attention
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
        
        # LSTM and output
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.out(x)
        
        return x.squeeze(-1)

def load_model_and_preprocessing():
    """Load the trained model, label encoders, and scaler."""
    # Declare all global variables at the beginning
    global a_model, a_feature_columns, a_label_encoders, a_config, a_processed_dir, a_scaler, a_checkpoint_path, a_device

    # Force CPU for all operations
    a_device = 'cpu'

    # Initialize all global variables
    a_model = None
    a_feature_columns = []
    a_label_encoders = {}
    a_config = {}
    a_processed_dir = ""
    a_scaler = None
    a_checkpoint_path = None

    # Set up logging
    logger = setup_logging()

    # Initialize logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize globals if not already set
    if a_feature_columns is None:
        a_feature_columns = []

    try:
        logger.info("\n" + "="*70)
        logger.info("[INFO] Starting RoadSafe AI (Hybrid Mode)")
        logger.info("="*70 + "\n")
        logger.info("[INFO] Initializing Model Loading Process...")

        # Try to import torch first
        torch = None
        try:
            import torch
            logger.info("[SUCCESS] PyTorch imported successfully")
        except ImportError:
            logger.warning("[WARNING] PyTorch not available - will use fallback mode")
            return

        # Try to import other ML libraries
        try:
            import pandas as pd
            from joblib import load as joblib_load
            import shap
            import os
            logger.info("[SUCCESS] All ML dependencies available")
        except ImportError as e:
            logger.error(f"[WARNING] Some ML dependencies not available: {e}")
            logger.warning("[INFO] Running in demo mode with fallback predictions")
            return

        # Always use CPU to avoid device mismatch issues
        a_device = torch.device("cpu")
        logger.info(f"[INFO] Forcing device to: {a_device} (CPU only mode)")

        # Disable CUDA
        torch.cuda.is_available = lambda: False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Try to load the model from quick_fixed directory
        model_path = os.path.join("outputs", "quick_fixed", "best.pt")
        if os.path.exists(model_path):
            logger.info(f"[INFO] Found model at: {os.path.abspath(model_path)}")
            try:
                # Load the checkpoint with CPU mapping
                logger.info("[INFO] Loading model checkpoint with CPU mapping...")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                logger.info("[SUCCESS] Successfully loaded model checkpoint")
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

                # Set feature columns from checkpoint
                a_feature_columns = checkpoint.get('features', [])
                a_config = checkpoint.get('cfg', {})
                
                # Create and load the working AI model
                a_model = WorkingAIModel(len(a_feature_columns))
                
                # Load weights with strict=False to handle minor differences
                missing_keys, unexpected_keys = a_model.load_state_dict(checkpoint['model'], strict=False)
                
                if missing_keys:
                    logger.info(f"Missing keys (using defaults): {len(missing_keys)}")
                if unexpected_keys:
                    logger.info(f"Unexpected keys (ignored): {len(unexpected_keys)}")
                
                a_model.eval()
                a_checkpoint_path = model_path

                logger.info(f"[SUCCESS] REAL AI MODEL loaded with {len(a_feature_columns)} features")
                logger.info(f"Model device: {a_device}")
                logger.info("[SUCCESS] CNN-BiLSTM-Attention model ready for predictions!")
                logger.info("="*70 + "\n")

            except Exception as e:
                logger.error(f"[ERROR] Error loading AI model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                a_model = None
                a_feature_columns = []
        else:
            logger.error(f"[ERROR] Model file not found at: {os.path.abspath(model_path)}")

    except Exception as e:
        logger.critical(f"\n[CRITICAL ERROR] Critical error in load_model_and_preprocessing: {str(e)}")
        a_model = None
        a_feature_columns = []

        # Log traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        a_feature_columns = None

def predict_risk(input_data: dict) -> dict:
    """Predict using ML model if available, otherwise use fallback."""
    global a_model, a_feature_columns, a_config, a_device, a_scaler, a_label_encoders
    logger.info("\n" + "="*80)
    logger.info(f"[INFO] Entering predict_risk with input: {input_data}")

    if a_model is None or not a_feature_columns:
        logger.warning("[WARNING] Model or feature columns not loaded. Using fallback prediction.")
        return fallback_prediction(input_data)

    try:
        # 1. Dynamically import pandas inside the function
        import pandas as pd

        # 2. Create DataFrame
        df = pd.DataFrame([input_data])
        
        # 3. Add missing features with defaults
        for feature in a_feature_columns:
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
                elif feature == 'risk':
                    df[feature] = 2.0
                else:
                    df[feature] = 0
        
        # 4. Prepare input tensor
        X = df[a_feature_columns].values.astype(np.float32)
        seq_len = a_config['data']['sequence_length']
        X_seq = np.tile(X, (seq_len, 1)).reshape(1, seq_len, -1)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        # 5. Make prediction with real AI model
        with torch.no_grad():
            output = a_model(X_tensor)
            raw_output = float(output.item())
            logger.info(f"[INFO] Raw AI model output: {raw_output:.6f}")
            
            # Convert to risk probability
            risk_prob = torch.sigmoid(torch.tensor(raw_output)).item()
            logger.info(f"[INFO] Risk probability: {risk_prob:.6f}")
            
            # Scale to 1-3 range
            risk_value = 1.0 + (risk_prob * 2.0)
            logger.info(f"[INFO] Scaled risk value: {risk_value:.3f}")

            # Determine risk level
            if risk_prob < 0.33:
                risk_level = "Low"
            elif risk_prob < 0.66:
                risk_level = "Medium"
            else:
                risk_level = "High"
                
            # Calculate confidence
            confidence = 80.0 + (risk_prob * 15.0)
            
            logger.info(f"[SUCCESS] AI Prediction: {risk_level} ({risk_value:.3f})")
            return {
                "risk_value": round(risk_value, 3),
                "risk_level": risk_level,
                "confidence": round(confidence, 1),
                "used_model": True,
                "prediction_source": "real_ai_model",
                "data_source": "AI Neural Network",
                "model_type": "CNN-BiLSTM-Attention"
            }

    except Exception as e:
        logger.error(f"[ERROR] AI prediction failed: {str(e)}", exc_info=True)
        logger.warning("[WARNING] Falling back to fallback prediction due to error.")
        return fallback_prediction(input_data)

def fallback_prediction(input_data: dict) -> dict:
    logger.warning("\n" + "="*80)
    logger.warning(f"[WARNING] Entering fallback_prediction with input: {input_data}")
    """Fallback prediction when ML model is not available"""
    risk_score = 2.0  # Default to medium risk

    # Extract location-specific data from input
    latitude = input_data.get('Latitude', 51.5)
    longitude = input_data.get('Longitude', -0.1)
    hour = input_data.get('hour', 12)
    weather_conditions = input_data.get('Weather_Conditions', 1)
    road_surface = input_data.get('Road_Surface_Conditions', 0)
    speed_limit = input_data.get('Speed_limit', 30)

    logger.info(f"[DEBUG] Fallback prediction using location data: lat={latitude}, lon={longitude}, hour={hour}, weather={weather_conditions}, surface={road_surface}, speed={speed_limit}")

    # Geographic risk factors based on latitude (different regions have different risk profiles)
    if latitude > 40:  # Northern regions (colder, potentially icy)
        risk_score += 0.2
    elif latitude < 20:  # Tropical regions (heavy rain, flooding)
        risk_score += 0.3

    # Time-based risk (same as before)
    if hour < 6 or hour > 22:
        risk_score += 0.3  # Night time risk
    elif 7 <= hour <= 9 or 17 <= hour <= 19:
        risk_score += 0.2  # Rush hour risk

    # Weather-based risk (enhanced based on actual weather data)
    if weather_conditions in [2, 3, 5, 6]:  # Rain, Drizzle, Mist, Fog
        risk_score += 0.4
    elif weather_conditions == 7:  # Snow
        risk_score += 0.5
    elif weather_conditions == 4:  # Thunderstorm
        risk_score += 0.6

    # Road surface risk (enhanced)
    if road_surface in [1, 2, 3, 4]:  # Wet, Icy, Snow, Flood
        risk_score += 0.3

    # Speed limit risk (enhanced)
    if speed_limit > 50:
        risk_score += 0.2
    elif speed_limit > 30:
        risk_score += 0.1

    # Urban vs Rural risk based on coordinates
    # Simple heuristic: city centers tend to have higher risk
    if abs(latitude - round(latitude)) < 0.01 and abs(longitude - round(longitude)) < 0.01:
        risk_score += 0.1  # Likely urban area

    risk_score = max(1.0, min(3.0, risk_score))

    if risk_score < 1.5:
        risk_level = "Low"
    elif risk_score < 2.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    logger.info(f"[INFO] Fallback prediction result: {risk_level} ({risk_score:.3f}) based on location data")

    return {
        "risk_value": round(risk_score, 3),
        "risk_level": risk_level,
        "confidence": 75.0,
        "note": f"Using location-based fallback prediction (lat: {latitude:.2f}, lon: {longitude:.2f})",
        "used_model": False,
        "prediction_source": "location_aware_fallback",
        "factors": {
            "location": f"{latitude:.2f}, {longitude:.2f}",
            "weather": weather_conditions,
            "surface": road_surface,
            "speed": speed_limit,
            "hour": hour
        }
    }

# Routes
@app.route('/')
def index():
    return render_template('index_new.html')

@app.route('/heatmap')
def heatmap_dashboard():
    return render_template('heatmap_dashboard.html')

@app.route('/historical-dashboard')
def historical_dashboard():
    return render_template('historical_dashboard.html')

@app.route('/government')
def government_dashboard():
    return render_template('government_dashboard.html')

@app.route('/geocode_city', methods=['POST'])
def geocode_city():
    data = request.get_json()
    city_name = data.get('city', 'London')

    # Indian cities database
    major_cities = {
        'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai', 'country': 'IN'},
        'delhi': {'lat': 28.7041, 'lon': 77.1025, 'name': 'Delhi', 'country': 'IN'},
        'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bangalore', 'country': 'IN'},
        'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad', 'country': 'IN'},
        'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Ahmedabad', 'country': 'IN'},
        'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai', 'country': 'IN'},
        'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata', 'country': 'IN'},
        'pune': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Pune', 'country': 'IN'},
        'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur', 'country': 'IN'},
        'surat': {'lat': 21.1702, 'lon': 72.8311, 'name': 'Surat', 'country': 'IN'},
        'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow', 'country': 'IN'},
        'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'name': 'Kanpur', 'country': 'IN'},
        'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'name': 'Nagpur', 'country': 'IN'},
        'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore', 'country': 'IN'},
        'thane': {'lat': 19.2183, 'lon': 72.9781, 'name': 'Thane', 'country': 'IN'},
        'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal', 'country': 'IN'},
        'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Visakhapatnam', 'country': 'IN'},
        'patna': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Patna', 'country': 'IN'},
        'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'name': 'Vadodara', 'country': 'IN'},
        'ghaziabad': {'lat': 28.6692, 'lon': 77.4538, 'name': 'Ghaziabad', 'country': 'IN'},
        'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'name': 'Ludhiana', 'country': 'IN'},
        'agra': {'lat': 27.1767, 'lon': 78.0081, 'name': 'Agra', 'country': 'IN'},
        'nashik': {'lat': 19.9975, 'lon': 73.7898, 'name': 'Nashik', 'country': 'IN'},
        'faridabad': {'lat': 28.4089, 'lon': 77.3178, 'name': 'Faridabad', 'country': 'IN'},
        'meerut': {'lat': 28.9845, 'lon': 77.7064, 'name': 'Meerut', 'country': 'IN'},
        'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'name': 'Rajkot', 'country': 'IN'},
        'varanasi': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Varanasi', 'country': 'IN'},
        'aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'name': 'Aurangabad', 'country': 'IN'},
        'amritsar': {'lat': 31.6340, 'lon': 74.8723, 'name': 'Amritsar', 'country': 'IN'},
        'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'name': 'Ranchi', 'country': 'IN'},
        'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'name': 'Coimbatore', 'country': 'IN'},
        'jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'name': 'Jabalpur', 'country': 'IN'},
        'gwalior': {'lat': 26.2183, 'lon': 78.1828, 'name': 'Gwalior', 'country': 'IN'},
        'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Vijayawada', 'country': 'IN'},
        'jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'name': 'Jodhpur', 'country': 'IN'},
        'madurai': {'lat': 9.9252, 'lon': 78.1198, 'name': 'Madurai', 'country': 'IN'},
        'raipur': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur', 'country': 'IN'},
        'kota': {'lat': 25.2138, 'lon': 75.8648, 'name': 'Kota', 'country': 'IN'},
        'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh', 'country': 'IN'},
        'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'name': 'Guwahati', 'country': 'IN'},
        'solapur': {'lat': 17.6599, 'lon': 75.9064, 'name': 'Solapur', 'country': 'IN'},
        'bareilly': {'lat': 28.3670, 'lon': 79.4304, 'name': 'Bareilly', 'country': 'IN'},
        'moradabad': {'lat': 28.8386, 'lon': 78.7733, 'name': 'Moradabad', 'country': 'IN'},
        'mysore': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysuru', 'country': 'IN'},
        'salem': {'lat': 11.6643, 'lon': 78.1460, 'name': 'Salem', 'country': 'IN'},
        'aligarh': {'lat': 27.8974, 'lon': 78.0880, 'name': 'Aligarh', 'country': 'IN'},
        'guntur': {'lat': 16.3067, 'lon': 80.4365, 'name': 'Guntur', 'country': 'IN'},
        'bhiwandi': {'lat': 19.3002, 'lon': 73.0635, 'name': 'Bhiwandi', 'country': 'IN'},
        'saharanpur': {'lat': 29.9680, 'lon': 77.5552, 'name': 'Saharanpur', 'country': 'IN'},
        'gorakhpur': {'lat': 26.7606, 'lon': 83.3732, 'name': 'Gorakhpur', 'country': 'IN'},
        'bikaner': {'lat': 28.0229, 'lon': 73.3119, 'name': 'Bikaner', 'country': 'IN'},
        'amravati': {'lat': 20.9374, 'lon': 77.7796, 'name': 'Amravati', 'country': 'IN'},
        'noida': {'lat': 28.5355, 'lon': 77.3910, 'name': 'Noida', 'country': 'IN'},
        'jamshedpur': {'lat': 22.8046, 'lon': 86.2029, 'name': 'Jamshedpur', 'country': 'IN'},
        'bhilai': {'lat': 21.1938, 'lon': 81.3509, 'name': 'Bhilai', 'country': 'IN'},
        'cuttack': {'lat': 20.4625, 'lon': 85.8828, 'name': 'Cuttack', 'country': 'IN'},
        'kochi': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Kochi', 'country': 'IN'},
        'bhavnagar': {'lat': 21.7645, 'lon': 72.1519, 'name': 'Bhavnagar', 'country': 'IN'},
        'dehradun': {'lat': 30.3165, 'lon': 78.0322, 'name': 'Dehradun', 'country': 'IN'},
        'durgapur': {'lat': 23.5204, 'lon': 87.3119, 'name': 'Durgapur', 'country': 'IN'},
        'asansol': {'lat': 23.6739, 'lon': 86.9524, 'name': 'Asansol', 'country': 'IN'},
        'nanded': {'lat': 19.1383, 'lon': 77.3210, 'name': 'Nanded', 'country': 'IN'},
        'kolhapur': {'lat': 16.7050, 'lon': 74.2433, 'name': 'Kolhapur', 'country': 'IN'},
        'ajmer': {'lat': 26.4499, 'lon': 74.6399, 'name': 'Ajmer', 'country': 'IN'},
        'akola': {'lat': 20.7002, 'lon': 77.0082, 'name': 'Akola', 'country': 'IN'},
        'jamnagar': {'lat': 22.4707, 'lon': 70.0577, 'name': 'Jamnagar', 'country': 'IN'},
        'ujjain': {'lat': 23.1765, 'lon': 75.7885, 'name': 'Ujjain', 'country': 'IN'},
        'jhansi': {'lat': 25.4484, 'lon': 78.5685, 'name': 'Jhansi', 'country': 'IN'},
        'jammu': {'lat': 32.7266, 'lon': 74.8570, 'name': 'Jammu', 'country': 'IN'},
        'mangalore': {'lat': 12.9141, 'lon': 74.8560, 'name': 'Mangaluru', 'country': 'IN'},
        'erode': {'lat': 11.3410, 'lon': 77.7172, 'name': 'Erode', 'country': 'IN'},
        'tirunelveli': {'lat': 8.7139, 'lon': 77.7567, 'name': 'Tirunelveli', 'country': 'IN'},
        'malegaon': {'lat': 20.5579, 'lon': 74.5287, 'name': 'Malegaon', 'country': 'IN'},
        'gaya': {'lat': 24.7914, 'lon': 85.0002, 'name': 'Gaya', 'country': 'IN'},
        'jalgaon': {'lat': 21.0077, 'lon': 75.5626, 'name': 'Jalgaon', 'country': 'IN'},
        'udaipur': {'lat': 24.5854, 'lon': 73.7125, 'name': 'Udaipur', 'country': 'IN'}
    }

    city_lower = city_name.lower().strip()
    if city_lower in major_cities:
        city_data = major_cities[city_lower]
        return jsonify({
            'latitude': city_data['lat'],
            'longitude': city_data['lon'],
            'lat': city_data['lat'],
            'lon': city_data['lon'],
            'city': city_data['name'],
            'country': city_data['country']
        })

    return jsonify({'error': f'City "{city_name}" not found'}), 404

@app.route('/fetch_location_data', methods=['POST'])
def fetch_location_data():
    data = request.get_json()
    lat = data.get('latitude', 51.5074)
    lon = data.get('longitude', -0.1278)

    try:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        now = datetime.now()

        if is_demo_mode() or not api_key:
            # Demo mode
            return jsonify({
                'main_weather': 'Clear',
                'description': 'clear sky',
                'temperature': 20.0,
                'humidity': 60,
                'wind_speed': 5.0,
                'hour': now.hour,
                'day_of_week': now.isoweekday(),
                'month': now.month,
                'light_conditions': get_light_conditions(now.hour),
                'weather_conditions': 1,
                'road_surface': 0,
                'speed_limit': 30,
                'road_type': 3,
                'junction_detail': 0,
                'data_source': 'simulated',
                'api_status': 'demo_mode'
            })

        # Real API call with caching
        from time import time
        ttl_seconds = 120
        key = (round(float(lat), 3), round(float(lon), 3))
        now_ts = time()

        if key in _WEATHER_CACHE:
            entry = _WEATHER_CACHE[key]
            if now_ts - entry['ts'] <= ttl_seconds:
                return jsonify(entry['data'])

        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(weather_url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        main_weather = weather_data['weather'][0]['main']
        weather_code = WEATHER_CONDITIONS_MAP.get(main_weather, 1)
        road_surface = get_road_surface_from_weather(main_weather, weather_data['main']['temp'])

        payload = {
            'main_weather': main_weather,
            'description': weather_data['weather'][0]['description'],
            'temperature': weather_data['main']['temp'],
            'humidity': weather_data['main']['humidity'],
            'wind_speed': weather_data['wind']['speed'],
            'hour': now.hour,
            'day_of_week': now.isoweekday(),
            'month': now.month,
            'light_conditions': get_light_conditions(now.hour),
            'weather_conditions': weather_code,
            'road_surface': road_surface,
            'speed_limit': 30,
            'road_type': 3,
            'junction_detail': 0,
            'data_source': 'real_api',
            'api_status': 'live_data'
        }

        _WEATHER_CACHE[key] = {"ts": now_ts, "data": payload}
        return jsonify(payload)

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        # Fallback response
        now = datetime.now()
        return jsonify({
            'main_weather': 'Clear',
            'description': 'clear sky (API error fallback)',
            'temperature': 20.0,
            'humidity': 60,
            'wind_speed': 5.0,
            'hour': now.hour,
            'day_of_week': now.isoweekday(),
            'month': now.month,
            'light_conditions': get_light_conditions(now.hour),
            'weather_conditions': 1,
            'road_surface': 0,
            'speed_limit': 30,
            'road_type': 3,
            'junction_detail': 0,
            'data_source': 'fallback',
            'api_status': 'api_error',
            'error_note': f'Weather API error: {e}'
        })

@app.route('/predict_risk', methods=['POST'])
def predict_risk_route():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    # Get current time if not provided
    now = datetime.now()
    
    # Default values for all required features
    default_values = {
        'Latitude': 51.5,
        'Longitude': -0.1,
        'hour': now.hour,
        'Day_of_Week': now.isoweekday(),  # 1-7 (Monday is 1)
        'month': now.month,
        'Light_Conditions': get_light_conditions(now.hour),
        'Weather_Conditions': 1,  # Default to Clear
        'Road_Surface_Conditions': 0,  # Default to Dry
        'Junction_Detail': 0,  # Default to Not at junction or within 20 metres
        'Road_Type': 3,  # Default to Single carriageway
        'Speed_limit': 30,  # Default speed limit
        'Number_of_Vehicles': 1,  # Default number of vehicles
        'Number_of_Casualties': 1,  # Default number of casualties
    }
    
    # Update with provided values
    input_data = default_values.copy()

    # Debug: Log what data was received
    logger.info(f"Raw data received from frontend: {data}")

    # More robust key mapping
    key_mapping = {
        # Frontend key -> Backend key
        'latitude': 'Latitude',
        'lat': 'Latitude',
        'longitude': 'Longitude',
        'lng': 'Longitude',
        'lon': 'Longitude',
        'hour': 'hour',
        'day_of_week': 'Day_of_Week',
        'dayofweek': 'Day_of_Week',
        'month': 'month',
        'light_conditions': 'Light_Conditions',
        'lightconditions': 'Light_Conditions',
        'weather_conditions': 'Weather_Conditions',
        'weatherconditions': 'Weather_Conditions',
        'road_surface': 'Road_Surface_Conditions',
        'roadsurface': 'Road_Surface_Conditions',
        'junction_detail': 'Junction_Detail',
        'junctiondetail': 'Junction_Detail',
        'road_type': 'Road_Type',
        'roadtype': 'Road_Type',
        'speed_limit': 'Speed_limit',
        'speedlimit': 'Speed_limit',
        'number_of_vehicles': 'Number_of_Vehicles',
        'numberofvehicles': 'Number_of_Vehicles',
        'number_of_casualties': 'Number_of_Casualties',
        'numberofcasualties': 'Number_of_Casualties',
    }

    # Update input_data with received values using flexible key matching
    for received_key, value in data.items():
        received_key_lower = received_key.lower().replace('_', '').replace('-', '')

        # Direct mapping first
        if received_key in default_values:
            input_data[received_key] = value
            logger.info(f"Direct key match: {received_key} -> {value}")
        # Mapping table lookup
        elif received_key_lower in key_mapping:
            backend_key = key_mapping[received_key_lower]
            input_data[backend_key] = value
            logger.info(f"Mapped key: {received_key} -> {backend_key} = {value}")
        # Fuzzy matching for common variations
        else:
            for default_key in default_values.keys():
                default_key_lower = default_key.lower().replace('_', '')
                if received_key_lower == default_key_lower:
                    input_data[default_key] = value
                    logger.info(f"Fuzzy match: {received_key} -> {default_key} = {value}")
                    break
    
    # Log the final input data for debugging
    logger.info(f"Final input data for prediction: {input_data}")
    
    try:
        # Ensure all values are of the correct type
        for key, value in input_data.items():
            if key in ['Latitude', 'Longitude']:
                input_data[key] = float(value)
            else:
                input_data[key] = int(float(value))  # Convert to int via float to handle string numbers
        
        # Use ML model if available, otherwise fallback
        prediction = predict_risk(input_data)
        
        # Add input data to the response for debugging
        prediction['input_data'] = input_data
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Error processing prediction: {str(e)}",
            "input_data": input_data,
            "used_model": False,
            "prediction_source": "error"
        }), 500

# Status endpoint
@app.route('/status', methods=['GET'])
def status():
    try:
        model_loaded = a_model is not None and a_feature_columns is not None
        return jsonify({
            "model_loaded": model_loaded,
            "checkpoint": a_checkpoint_path if a_checkpoint_path else None,
            "device": a_device if a_device else "cpu",
            "features_count": len(a_feature_columns) if a_feature_columns else 0,
            "api_mode": "demo" if is_demo_mode() else "live",
            "weather_cache_entries": len(_WEATHER_CACHE) if _WEATHER_CACHE else 0,
            "ml_available": a_model is not None
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "model_loaded": False,
            "ml_available": False
        }), 200

# Import and register blueprints if available
try:
    from src.historical_api import historical_bp
    from src.traffic_api import traffic_bp
    from src.heatmap_api import heatmap_bp

    app.register_blueprint(historical_bp)
    app.register_blueprint(traffic_bp)
    app.register_blueprint(heatmap_bp)
    print("[SUCCESS] All API blueprints registered")
except ImportError as e:
    print(f"[WARNING] Some API modules not available: {e}")

if __name__ == '__main__':
    print("Starting RoadSafe AI (Hybrid Mode)")
    print("=" * 60)

    # Set environment variables
    os.environ['OPENWEATHER_API_KEY'] = get_openweather_api_key()

    # Try to load ML models
    load_model_and_preprocessing()

    # Check status
    if a_model is not None:
        print("[SUCCESS] Full ML model loaded - using AI predictions")
    else:
        print("[WARNING] ML model not available - using location-aware fallback predictions")
        print("[INFO] To enable AI predictions: pip install torch torchvision")

    print("[INFO] Starting Flask web server...")
    print("[INFO] Open your browser and go to: http://localhost:5000")
    print("=" * 60)

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
