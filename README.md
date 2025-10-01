# Road Traffic Accident Risk Prediction 🚗⚠️

[![Implementation Status](https://img.shields.io/badge/Status-Complete-brightgreen)](IMPLEMENTATION_SUMMARY.md)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/10234567)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-ready web application implementing the IEEE Access paper **"Road Traffic Accident Risk Prediction and Key Factor Identification Framework Based on Explainable Deep Learning"** with real-time weather integration and interactive visualization.

## 🎯 Key Features

### 🧠 **Deep Learning Architecture**
- **CNN-BiLSTM-Attention Model**: Spatiotemporal feature extraction with local attention mechanisms
- **DeepSHAP Explainability**: Global and per-instance feature importance analysis
- **Robust Training**: SMOTE+UnderSampler for class imbalance, weighted loss, regularization

### 🌐 **Real-time Web Interface**
- **Interactive Map**: Click-to-predict with live weather data
- **OpenWeatherMap Integration**: Real weather conditions with 2-minute caching
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Fallback Support**: Simulated data when API unavailable

### 🔧 **Production Features**
- **API Endpoints**: RESTful services for prediction and explainability
- **Error Handling**: Global JSON error handlers, graceful degradation
- **Monitoring**: Status endpoint with system diagnostics
- **Testing**: Comprehensive test suite with pytest
- **Security**: Environment-based API key management

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone repository
git clone <repository-url>
cd road

# Install dependencies
pip install -r requirements.txt
```

### 2. **API Configuration**
```bash
# Create .env file with your OpenWeatherMap API key
echo "OPENWEATHER_API_KEY=your_api_key_here" > .env

# Or use the setup script
python setup_api_key.py
```

### 3. **Run Application**
```bash
# Start Flask development server
python app.py

# Access at: http://127.0.0.1:5000
```

### 4. **Verify Installation**
```bash
# Run verification tests
python scripts/verify_implementation.py

# Run API tests
pytest tests/ -v
```

## 📊 Model Performance

Based on IEEE paper benchmarks:
- **UK Dataset MAE**: 0.2475 (target from paper)
- **US Dataset MAE**: 0.2683 (target from paper)
- **Precision/Recall/F1**: 0.80+ range for balanced evaluation

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/status` | GET | System diagnostics |
| `/predict_risk` | POST | Accident risk prediction |
| `/fetch_location_data` | POST | Weather + location data |
| `/geocode_city` | POST | City to coordinates |
| `/explain/global` | GET | Global SHAP ranking |
| `/explain/instance` | POST | Per-instance SHAP |

### Example Usage
```bash
# Get system status
curl http://localhost:5000/status

# Predict risk
curl -X POST http://localhost:5000/predict_risk \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 51.5074,
    "longitude": -0.1278,
    "hour": 14,
    "day_of_week": 3,
    "weather_conditions": 1,
    "road_surface": 0,
    "speed_limit": 30
  }'
```

## 🧪 Advanced Usage

### **Generate SHAP Analysis & Retrain Top-15 Features**
```bash
python scripts/run_top15.py \
  --checkpoint outputs/uk_full_e100/best.pt \
  --config config/uk_config.yaml \
  --topk 15 --epochs 50
```

### **Run Full Benchmark Suite**
```bash
python scripts/run_benchmarks.py \
  --config config/uk_config.yaml \
  --epochs 50 --tag benchmark_uk
```

### **Preprocess Dataset with Paper Settings**
```bash
python scripts/dataset_preprocessing.py \
  --dataset UK --config config/uk_config.yaml
```

## 🔬 Paper Implementation Compliance

### ✅ **Model Architecture**
- [x] CNN for spatial feature extraction
- [x] BiLSTM for temporal feature extraction
- [x] Local spatial attention mechanism
- [x] Local temporal attention mechanism
- [x] Regularization (dropout, batch normalization)

### ✅ **Data Processing**
- [x] Missing value filtering (10% threshold)
- [x] Near-zero variance filtering (2% threshold)
- [x] SMOTE oversampling + Random undersampling
- [x] Chronological train/val/test splits (60/20/20)
- [x] 32 UK features from Table 1

### ✅ **Evaluation & Explainability**
- [x] MAE, Precision, Recall, F1-Score metrics
- [x] DeepSHAP global feature ranking
- [x] Per-severity level factor analysis
- [x] Top-15 feature retraining validation

## 🌍 Real-time Capabilities

- **Live Weather Data**: OpenWeatherMap API integration
- **Geocoding**: City name to coordinates conversion
- **Caching**: 2-minute TTL to reduce API calls
- **Fallback**: Simulated data when API unavailable
- **Global Coverage**: Supports worldwide locations

## 📈 Deployment Options

### **Development**
```bash
python app.py  # Flask development server
```

### **Production**
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### **Docker**
```bash
docker build -t road-risk-prediction .
docker run -p 5000:5000 -e OPENWEATHER_API_KEY=your_key road-risk-prediction
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for comprehensive production setup.

## 📚 Documentation

- 📖 **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: Complete technical details
- 🚀 **[Deployment Guide](DEPLOYMENT_GUIDE.md)**: Production deployment instructions
- 🧪 **[API Documentation](docs/API.md)**: Endpoint specifications
- 🔬 **[Model Architecture](docs/MODEL.md)**: Deep learning implementation details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IEEE Access Paper**: Original research framework
- **OpenWeatherMap**: Real-time weather data
- **PyTorch**: Deep learning framework
- **Flask**: Web application framework
- **SHAP**: Model explainability library

---

**🎯 Ready to predict accident risks with state-of-the-art deep learning!** 🚗⚠️# Road-Accident-Risk-Prediction
# FINAL-ROAD-ACCIDENT-RISK-PREDICTION
