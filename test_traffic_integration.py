#!/usr/bin/env python3
"""
Test script to verify traffic API integration
Shows the difference between demo and real traffic data
"""
import requests
import json
from datetime import datetime

def test_traffic_api():
    """Test traffic API endpoints"""
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("🚦 Traffic API Integration Test")
    print("=" * 60)
    
    # Test 1: Check available providers
    print("\n1️⃣ Checking available traffic providers...")
    try:
        response = requests.get(f"{base_url}/api/traffic/providers")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['providers'])} provider(s):")
            for provider in data['providers']:
                print(f"   - {provider['display_name']}: {provider['name']}")
                if provider.get('note'):
                    print(f"     Note: {provider['note']}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Get current traffic for London
    print("\n2️⃣ Getting traffic data for London (51.5074, -0.1278)...")
    try:
        response = requests.get(f"{base_url}/api/traffic/current?lat=51.5074&lon=-0.1278")
        if response.status_code == 200:
            data = response.json()
            traffic = data['traffic']
            risk = data['risk_analysis']
            
            print(f"✅ Traffic data received:")
            print(f"   Provider: {traffic['provider']}")
            print(f"   Congestion: {traffic['congestion_percentage']}%")
            print(f"   Average Speed: {traffic['average_speed_kmh']} km/h")
            print(f"   Free Flow Speed: {traffic['free_flow_speed_kmh']} km/h")
            print(f"   Incidents: {traffic['incidents_count']}")
            print(f"   Confidence: {traffic['confidence'] * 100}%")
            print(f"   Risk Level: {risk['risk_level']}")
            print(f"   Risk Factor: {risk['traffic_risk_factor']}")
            
            if traffic['provider'] == 'demo':
                print("\n⚠️  Using DEMO data (simulated)")
                print("   To use real traffic data:")
                print("   1. Run: python setup_traffic_api.py")
                print("   2. Get a free API key from TomTom or HERE")
                print("   3. Restart the Flask app")
            else:
                print(f"\n✅ Using REAL traffic data from {traffic['provider'].upper()}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Test multiple locations
    print("\n3️⃣ Testing traffic at multiple locations...")
    locations = [
        ("London", 51.5074, -0.1278),
        ("Manchester", 53.4808, -2.2426),
        ("Birmingham", 52.4862, -1.8904),
        ("Mumbai", 19.0760, 72.8777),
        ("New York", 40.7128, -74.0060)
    ]
    
    for city, lat, lon in locations:
        try:
            response = requests.get(f"{base_url}/api/traffic/current?lat={lat}&lon={lon}")
            if response.status_code == 200:
                data = response.json()
                traffic = data['traffic']
                risk = data['risk_analysis']
                print(f"   {city}: Congestion {traffic['congestion_percentage']}%, Risk: {risk['risk_level']}")
        except:
            print(f"   {city}: Failed to fetch")
    
    # Test 4: Enhanced risk with traffic
    print("\n4️⃣ Testing enhanced risk prediction (AI + Traffic)...")
    try:
        payload = {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "base_risk": 0.5  # Simulated base risk from AI model
        }
        response = requests.post(f"{base_url}/api/traffic/enhanced-risk", json=payload)
        if response.status_code == 200:
            data = response.json()
            risk = data['risk_analysis']
            print(f"✅ Enhanced risk calculation:")
            print(f"   Base Risk (AI): {risk['base_risk']}")
            print(f"   Traffic Risk: {risk['traffic_risk']}")
            print(f"   Combined Risk: {risk['enhanced_risk']}")
            print(f"   Risk Level: {risk['risk_level']}")
            print(f"   Traffic Impact: {risk['traffic_impact']}%")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Route analysis
    print("\n5️⃣ Testing route analysis...")
    try:
        # Sample route from London to Cambridge
        route = [
            {"lat": 51.5074, "lon": -0.1278},  # London
            {"lat": 51.7520, "lon": -0.3360},  # St Albans
            {"lat": 51.8959, "lon": -0.4909},  # Luton
            {"lat": 52.2053, "lon": 0.1218}    # Cambridge
        ]
        
        response = requests.post(f"{base_url}/api/traffic/route-analysis", json={"route": route})
        if response.status_code == 200:
            data = response.json()
            summary = data['route_summary']
            print(f"✅ Route analysis complete:")
            print(f"   Total Points: {summary['total_points']}")
            print(f"   Average Risk: {summary['average_risk']}")
            print(f"   Risk Level: {summary['risk_level']}")
            print(f"   High Risk Segments: {summary['high_risk_segments']}")
            
            if data.get('recommendations'):
                print("   Recommendations:")
                for rec in data['recommendations'][:3]:
                    print(f"   - {rec}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    # Final check
    try:
        response = requests.get(f"{base_url}/api/traffic/providers")
        if response.status_code == 200:
            data = response.json()
            providers = [p['name'] for p in data['providers'] if p['name'] != 'demo']
            
            if providers:
                print("✅ Real traffic providers configured:", ", ".join(providers))
                print("🎉 Your application is using REAL traffic data!")
            else:
                print("⚠️  No real traffic providers configured")
                print("📝 To add real traffic data:")
                print("   1. Run: python setup_traffic_api.py")
                print("   2. Follow the instructions to get a free API key")
                print("   3. Restart your Flask application")
    except:
        print("❌ Could not connect to Flask app. Is it running?")
    
    print("=" * 60)

if __name__ == "__main__":
    test_traffic_api()
