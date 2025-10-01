#!/usr/bin/env python3
"""
Heatmap Data Generator - Creates realistic accident data for visualization
"""

import json
import random
from datetime import datetime, timedelta

def generate_indian_accident_data():
    """Generate realistic accident data for Indian cities"""
    
    # Major Indian cities with coordinates and risk factors
    indian_cities = [
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "risk_factor": 3.2},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "risk_factor": 3.0},
        {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946, "risk_factor": 2.7},
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "risk_factor": 2.8},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "risk_factor": 2.9},
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "risk_factor": 2.6},
        {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "risk_factor": 2.5},
        {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "risk_factor": 2.5},
        {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "risk_factor": 2.4},
        {"name": "Surat", "lat": 21.1702, "lon": 72.8311, "risk_factor": 2.3},
        {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "risk_factor": 2.2},
        {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319, "risk_factor": 2.4},
        {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "risk_factor": 2.3},
        {"name": "Indore", "lat": 22.7196, "lon": 75.8577, "risk_factor": 2.2},
        {"name": "Thane", "lat": 19.2183, "lon": 72.9781, "risk_factor": 2.4},
        {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126, "risk_factor": 2.1},
        {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "risk_factor": 2.0},
        {"name": "Pimpri-Chinchwad", "lat": 18.6298, "lon": 73.7997, "risk_factor": 2.3},
        {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "risk_factor": 2.2},
        {"name": "Vadodara", "lat": 22.3072, "lon": 73.1812, "risk_factor": 2.1},
    ]
    
    accident_data = []
    
    # Generate accidents for each city
    for city in indian_cities:
        # Number of accidents based on city risk factor
        num_accidents = int(city["risk_factor"] * 50)  # 100-160 accidents per city
        
        for _ in range(num_accidents):
            # Random location within city bounds (±0.1 degrees)
            lat = city["lat"] + random.uniform(-0.1, 0.1)
            lon = city["lon"] + random.uniform(-0.1, 0.1)
            
            # Random time in last 30 days
            days_ago = random.randint(0, 30)
            hour = random.randint(0, 23)
            accident_time = datetime.now() - timedelta(days=days_ago, hours=hour)
            
            # Severity based on time and location
            if 22 <= hour or hour <= 5:  # Night
                severity = random.choices([1, 2, 3], weights=[20, 50, 30])[0]
            elif 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hour
                severity = random.choices([1, 2, 3], weights=[30, 50, 20])[0]
            else:  # Day
                severity = random.choices([1, 2, 3], weights=[50, 40, 10])[0]
            
            accident_data.append({
                "id": len(accident_data) + 1,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "severity": severity,
                "timestamp": accident_time.isoformat(),
                "city": city["name"],
                "casualties": random.randint(1, severity + 2),
                "vehicles": random.randint(1, 3),
                "weather": random.choices([1, 2, 3], weights=[70, 25, 5])[0],  # Mostly clear
                "road_type": random.randint(1, 6)
            })
    
    return accident_data

def generate_global_accident_data():
    """Generate accident data for global cities"""
    
    global_cities = [
        {"name": "London", "lat": 51.5074, "lon": -0.1278, "risk_factor": 2.8},
        {"name": "New York", "lat": 40.7128, "lon": -74.0060, "risk_factor": 3.1},
        {"name": "Dubai", "lat": 25.2048, "lon": 55.2708, "risk_factor": 2.9},
        {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "risk_factor": 2.7},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "risk_factor": 2.6},
        {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "risk_factor": 2.5},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "risk_factor": 2.3},
        {"name": "Toronto", "lat": 43.6532, "lon": -79.3832, "risk_factor": 2.4},
    ]
    
    accident_data = []
    
    for city in global_cities:
        num_accidents = int(city["risk_factor"] * 40)  # Fewer accidents for demo
        
        for _ in range(num_accidents):
            lat = city["lat"] + random.uniform(-0.05, 0.05)
            lon = city["lon"] + random.uniform(-0.05, 0.05)
            
            days_ago = random.randint(0, 30)
            hour = random.randint(0, 23)
            accident_time = datetime.now() - timedelta(days=days_ago, hours=hour)
            
            severity = random.choices([1, 2, 3], weights=[40, 45, 15])[0]
            
            accident_data.append({
                "id": len(accident_data) + 1,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "severity": severity,
                "timestamp": accident_time.isoformat(),
                "city": city["name"],
                "casualties": random.randint(1, severity + 1),
                "vehicles": random.randint(1, 2),
                "weather": random.choices([1, 2, 3], weights=[80, 15, 5])[0],
                "road_type": random.randint(1, 6)
            })
    
    return accident_data