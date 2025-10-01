#!/usr/bin/env python3
"""
Sample Data Generator for Historical Analytics Testing
Generates realistic sample accident and weather data for testing the historical dashboard
"""
import sys
import os
from datetime import datetime, timedelta
import random
import requests
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.historical_service import HistoricalDataService

def generate_sample_accident_data(num_records: int = 1000) -> List[Dict[str, Any]]:
    """Generate sample accident data for testing"""
    accidents = []

    # Base location (London area)
    base_lat, base_lon = 51.5074, -0.1278

    # Weather conditions and their frequencies
    weather_conditions = [
        ('Clear', 0.4), ('Rain', 0.25), ('Clouds', 0.2),
        ('Fog', 0.1), ('Snow', 0.05)
    ]

    road_types = ['Urban', 'Rural', 'Motorway', 'Single carriageway']
    severity_levels = [1, 2, 3]  # 1=minor, 2=moderate, 3=severe

    print(f"Generating {num_records} sample accident records...")

    for i in range(num_records):
        # Generate random date within the last 2 years
        days_back = random.randint(1, 730)
        accident_date = datetime.now() - timedelta(days=days_back)

        # Add some time clustering (more accidents during rush hours)
        hour = random.choices(
            [7, 8, 9, 17, 18, 19] + list(range(24)),
            weights=[0.15, 0.2, 0.15, 0.15, 0.2, 0.15] + [0.02] * 18
        )[0]

        accident_date = accident_date.replace(hour=hour, minute=random.randint(0, 59))

        # Generate location with some clustering
        lat_offset = random.gauss(0, 0.05)  # Cluster around London
        lon_offset = random.gauss(0, 0.05)

        # Select weather condition
        weather = random.choices(
            [w[0] for w in weather_conditions],
            weights=[w[1] for w in weather_conditions]
        )[0]

        # Determine road surface based on weather
        road_surface = 'Dry'
        if weather in ['Rain', 'Snow']:
            road_surface = 'Wet' if random.random() > 0.3 else 'Dry'

        accident = {
            'latitude': base_lat + lat_offset,
            'longitude': base_lon + lon_offset,
            'accident_date': accident_date,
            'severity': random.choice(severity_levels),
            'casualties': random.randint(0, 3),
            'vehicles_involved': random.randint(1, 4),
            'road_type': random.choice(road_types),
            'junction_detail': random.randint(0, 8),
            'weather_conditions': weather,
            'road_surface': road_surface,
            'light_conditions': 1 if 6 <= hour <= 18 else 0,  # Daylight vs dark
            'speed_limit': random.choice([20, 30, 40, 50, 60, 70]),
            'urban_rural': 'Urban' if random.random() > 0.3 else 'Rural',
            'police_force': 'Metropolitan Police',
            'local_authority': 'London'
        }

        accidents.append(accident)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_records} accident records...")

    return accidents

def generate_sample_weather_data(num_records: int = 500) -> List[Dict[str, Any]]:
    """Generate sample weather data for testing"""
    weather_data = []

    # Base location (London area)
    base_lat, base_lon = 51.5074, -0.1278

    # Weather patterns by month (seasonal variation)
    monthly_temps = {
        1: (5, 8),    # January: cold
        2: (6, 9),    # February
        3: (8, 12),   # March
        4: (10, 15),  # April
        5: (13, 18),  # May
        6: (16, 21),  # June
        7: (18, 23),  # July: warm
        8: (18, 22),  # August
        9: (15, 19),  # September
        10: (12, 16), # October
        11: (8, 12),  # November
        12: (6, 9)    # December: cold
    }

    print(f"Generating {num_records} sample weather records...")

    for i in range(num_records):
        # Generate random date within the last year
        days_back = random.randint(1, 365)
        weather_date = datetime.now() - timedelta(days=days_back)

        month = weather_date.month
        temp_range = monthly_temps[month]

        # Generate weather data
        temperature = round(random.uniform(temp_range[0], temp_range[1]), 1)
        humidity = random.randint(40, 90)
        wind_speed = round(random.uniform(0, 15), 1)

        # Weather main category
        if temperature < 0:
            weather_main = 'Snow'
        elif temperature < 5:
            weather_main = random.choice(['Snow', 'Rain', 'Clouds'])
        elif temperature < 15:
            weather_main = random.choice(['Rain', 'Clouds', 'Clear'])
        else:
            weather_main = random.choice(['Clear', 'Clouds', 'Rain'])

        weather_record = {
            'latitude': base_lat + random.gauss(0, 0.02),
            'longitude': base_lon + random.gauss(0, 0.02),
            'weather_date': weather_date,
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': round(random.uniform(0, 10), 1) if weather_main == 'Rain' else 0,
            'wind_speed': wind_speed,
            'weather_main': weather_main,
            'weather_description': f'{weather_main.lower()} conditions',
            'visibility': round(random.uniform(5, 20), 1),
            'pressure': round(random.uniform(990, 1030), 1)
        }

        weather_data.append(weather_record)

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_records} weather records...")

    return weather_data

def main():
    """Main function to load sample data"""
    print("🚀 Starting sample data generation for historical analytics...")
    print("=" * 60)

    # Initialize historical service
    historical_service = HistoricalDataService()

    try:
        # Generate and load accident data
        print("\n📊 Generating accident data...")
        accident_data = generate_sample_accident_data(1000)

        print("💾 Loading accident data into database...")
        accident_result = historical_service.ingest_accident_data(accident_data)

        if accident_result["success"]:
            print(f"✅ {accident_result['records_added']} accident records loaded successfully")
        else:
            print(f"❌ Error loading accident data: {accident_result['error']}")
            return

        # Generate and load weather data
        print("\n🌤️ Generating weather data...")
        weather_data = generate_sample_weather_data(500)

        print("💾 Loading weather data into database...")
        weather_result = historical_service.ingest_weather_data(weather_data)

        if weather_result["success"]:
            print(f"✅ {weather_result['records_added']} weather records loaded successfully")
        else:
            print(f"❌ Error loading weather data: {weather_result['error']}")
            return

        print("\n" + "=" * 60)
        print("🎉 Sample data generation completed successfully!")
        print("=" * 60)
        print("📈 You can now test the historical dashboard at:")
        print("🔗 http://localhost:5000/historical-dashboard")
        print("=" * 60)

        # Test the API endpoints
        print("\n🧪 Testing API endpoints...")

        # Test statistics endpoint
        try:
            response = requests.get('http://localhost:5000/api/historical/statistics?lat=51.5074&lon=-0.1278')
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Statistics endpoint working: {data['total_accidents']} accidents found")
            else:
                print(f"❌ Statistics endpoint error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing statistics endpoint: {e}")

        # Test peak hours endpoint
        try:
            response = requests.get('http://localhost:5000/api/historical/peak-hours?lat=51.5074&lon=-0.1278')
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Peak hours endpoint working: {len(data['peak_hours'])} peak hours identified")
            else:
                print(f"❌ Peak hours endpoint error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing peak hours endpoint: {e}")

        print("\n🎯 Historical analytics system is ready!")

    except Exception as e:
        print(f"❌ Error during sample data generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        historical_service.close()

if __name__ == "__main__":
    main()
