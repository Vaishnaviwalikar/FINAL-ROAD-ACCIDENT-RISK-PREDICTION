# Deployment Guide - Road Traffic Accident Risk Prediction

## 🚀 Production Deployment Options

### Option 1: Local Development Server
```bash
# Quick start for development/demo
python app.py
# Access: http://127.0.0.1:5000
```

### Option 2: Production WSGI Server
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
# Access: http://your-server:8000
```

### Option 3: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t road-risk-prediction .
docker run -p 5000:5000 -e OPENWEATHER_API_KEY=your_key road-risk-prediction
```

## 🔧 Environment Configuration

### Required Environment Variables
```bash
# .env file
OPENWEATHER_API_KEY=your_openweathermap_api_key_here
FLASK_ENV=production
FLASK_DEBUG=False
```

### Optional Configuration
```bash
# Database (if extending)
DATABASE_URL=postgresql://user:pass@localhost/roadrisk

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/roadrisk.log

# Cache settings
REDIS_URL=redis://localhost:6379/0
```

## 📊 Monitoring & Health Checks

### Health Check Endpoint
```bash
# Check system status
curl http://your-server:5000/status

# Expected response:
{
  "model_loaded": true,
  "checkpoint": "outputs/final_fixed/best.pt",
  "device": "cpu",
  "features_count": 28,
  "api_mode": "live",
  "weather_cache_entries": 5
}
```

### Monitoring Metrics
- Model load status
- API response times
- Weather API call frequency
- Cache hit rates
- Error rates by endpoint

## 🔒 Security Considerations

### API Key Security
- Never commit API keys to version control
- Use environment variables or secure vaults
- Rotate keys regularly
- Monitor API usage

### Input Validation
- All endpoints validate input parameters
- Coordinate bounds checking
- Rate limiting on prediction endpoints
- CORS configuration for web deployment

### Error Handling
- Global JSON error handlers prevent information leakage
- Structured logging for security monitoring
- Graceful degradation when services unavailable

## 📈 Performance Optimization

### Caching Strategy
- Weather data cached for 2 minutes
- Model predictions can be cached by input hash
- Static assets served with appropriate headers

### Model Optimization
- Use CPU-optimized PyTorch builds
- Consider ONNX conversion for faster inference
- Batch predictions when possible

### Database Considerations
- Index frequently queried columns
- Use connection pooling
- Consider read replicas for scaling

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy Road Risk Prediction

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
      - name: Verify implementation
        run: python scripts/verify_implementation.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Your deployment commands here
          echo "Deploying to production..."
```

## 🌐 Web Server Configuration

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /path/to/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### SSL/HTTPS Setup
```bash
# Using Certbot for Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

## 📱 Mobile Responsiveness

The web interface is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Mobile devices (iOS Safari, Android Chrome)
- Tablets (iPad, Android tablets)

## 🔍 Troubleshooting Production Issues

### Common Problems

1. **Model Not Loading**
   ```bash
   # Check model files exist
   ls -la outputs/*/best.pt
   
   # Check permissions
   chmod 644 outputs/*/best.pt
   
   # Verify with status endpoint
   curl http://localhost:5000/status
   ```

2. **Weather API Failures**
   ```bash
   # Test API key
   curl "http://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_KEY"
   
   # Check rate limits
   # OpenWeatherMap: 1000 calls/day free tier
   ```

3. **High Memory Usage**
   ```bash
   # Monitor memory usage
   ps aux | grep python
   
   # Consider model quantization
   # Use smaller batch sizes
   # Enable garbage collection
   ```

### Log Analysis
```bash
# Check application logs
tail -f /var/log/roadrisk.log

# Monitor error patterns
grep ERROR /var/log/roadrisk.log | tail -20

# Check API response times
grep "predict_risk" /var/log/roadrisk.log | awk '{print $NF}'
```

## 📊 Scaling Considerations

### Horizontal Scaling
- Load balancer (Nginx, HAProxy)
- Multiple app instances
- Shared cache (Redis)
- Database clustering

### Vertical Scaling
- Increase CPU/RAM
- GPU acceleration for model inference
- SSD storage for faster I/O

### Microservices Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Web UI    │    │  API Gateway│    │  Prediction │
│   Service   │◄──►│   Service   │◄──►│   Service   │
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   Weather   │
                   │   Service   │
                   └─────────────┘
```

## 🔄 Backup & Recovery

### Data Backup
```bash
# Backup model files
tar -czf models_backup_$(date +%Y%m%d).tar.gz outputs/

# Backup configuration
cp -r config/ config_backup_$(date +%Y%m%d)/
```

### Disaster Recovery
- Automated backups to cloud storage
- Infrastructure as Code (Terraform/CloudFormation)
- Database replication
- Multi-region deployment

## 📈 Analytics & Insights

### Usage Metrics
- Prediction requests per day
- Geographic distribution of queries
- Most common risk factors
- Model accuracy over time

### Business Intelligence
- Risk pattern analysis
- Seasonal trends
- Location-based insights
- Feature importance evolution

## 🎯 Next Steps for Production

1. **Set up monitoring** (Prometheus + Grafana)
2. **Configure logging** (ELK stack or similar)
3. **Implement rate limiting** (Redis-based)
4. **Add user authentication** (if required)
5. **Set up automated backups**
6. **Configure SSL certificates**
7. **Implement CI/CD pipeline**
8. **Performance testing** (load testing)
9. **Security audit** (penetration testing)
10. **Documentation** (API docs, runbooks)

This deployment guide ensures your road traffic accident risk prediction system is production-ready with proper monitoring, security, and scalability considerations.
