#!/usr/bin/env python3
"""
Quick setup script for Traffic API integration
Helps users get started with real traffic data
"""
import os
import sys
import requests
import webbrowser
from typing import Optional

def test_tomtom_api(api_key: str) -> bool:
    """Test if TomTom API key is valid"""
    try:
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            'key': api_key,
            'point': '51.5074,-0.1278',
            'unit': 'KMPH'
        }
        response = requests.get(url, params=params, timeout=5)
        return response.status_code == 200
    except:
        return False

def test_here_api(api_key: str) -> bool:
    """Test if HERE API key is valid"""
    try:
        url = "https://traffic.ls.hereapi.com/traffic/6.3/flow.json"
        params = {
            'apiKey': api_key,
            'prox': '51.5074,-0.1278,1000'
        }
        response = requests.get(url, params=params, timeout=5)
        return response.status_code == 200
    except:
        return False

def update_env_file(provider: str, api_key: str):
    """Update .env file with API key"""
    env_path = '.env'
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update the appropriate line
    key_mapping = {
        'tomtom': 'TOMTOM_API_KEY',
        'here': 'HERE_API_KEY',
        'mapbox': 'MAPBOX_API_KEY',
        'google': 'GOOGLE_MAPS_API_KEY'
    }
    
    env_var = key_mapping.get(provider.lower())
    if not env_var:
        return False
    
    # Find and update the line
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f'{env_var}='):
            lines[i] = f'{env_var}={api_key}\n'
            updated = True
            break
    
    # Write back
    if updated:
        with open(env_path, 'w') as f:
            f.writelines(lines)
        return True
    return False

def main():
    print("=" * 60)
    print("🚦 Traffic API Setup Assistant")
    print("=" * 60)
    print("\nThis script will help you set up real traffic data.")
    print("\nAvailable providers (all have free tiers):")
    print("1. TomTom - Easiest setup, 2,500 requests/day free")
    print("2. HERE - More data, 250,000 requests/month free")
    print("3. MapBox - Good for maps, 100,000 requests/month free")
    print("4. Google - Most comprehensive, $200 credit/month")
    print("5. Skip - Continue with demo data")
    
    choice = input("\nSelect provider (1-5): ").strip()
    
    if choice == '1':
        print("\n📍 Setting up TomTom Traffic API")
        print("-" * 40)
        print("Steps to get your free API key:")
        print("1. Opening TomTom Developer portal...")
        webbrowser.open("https://developer.tomtom.com/user/register")
        print("2. Sign up for a free account")
        print("3. Go to Dashboard → Apps → Create App")
        print("4. Copy your API key")
        print("-" * 40)
        
        api_key = input("\nPaste your TomTom API key here: ").strip()
        
        if api_key:
            print("\n🧪 Testing API key...")
            if test_tomtom_api(api_key):
                print("✅ API key is valid!")
                
                if update_env_file('tomtom', api_key):
                    print("✅ Updated .env file with TomTom API key")
                    print("\n🎉 Success! Real traffic data is now enabled!")
                    print("🔄 Please restart your Flask application to use real traffic data")
                    
                    # Test the integration
                    print("\n🧪 Testing traffic data fetch...")
                    test_traffic_integration()
                else:
                    print("❌ Failed to update .env file")
            else:
                print("❌ API key appears to be invalid. Please check and try again.")
        else:
            print("⚠️ No API key provided. Continuing with demo data.")
    
    elif choice == '2':
        print("\n📍 Setting up HERE Traffic API")
        print("-" * 40)
        print("Steps to get your free API key:")
        print("1. Opening HERE Developer portal...")
        webbrowser.open("https://developer.here.com/sign-up")
        print("2. Sign up for a free account")
        print("3. Create a new project")
        print("4. Generate REST API key")
        print("-" * 40)
        
        api_key = input("\nPaste your HERE API key here: ").strip()
        
        if api_key:
            print("\n🧪 Testing API key...")
            if test_here_api(api_key):
                print("✅ API key is valid!")
                
                if update_env_file('here', api_key):
                    print("✅ Updated .env file with HERE API key")
                    print("\n🎉 Success! Real traffic data is now enabled!")
                    print("🔄 Please restart your Flask application to use real traffic data")
                    
                    test_traffic_integration()
                else:
                    print("❌ Failed to update .env file")
            else:
                print("❌ API key appears to be invalid. Please check and try again.")
        else:
            print("⚠️ No API key provided. Continuing with demo data.")
    
    elif choice == '3':
        print("\n📍 Setting up MapBox API")
        print("-" * 40)
        print("1. Opening MapBox...")
        webbrowser.open("https://account.mapbox.com/auth/signup/")
        print("2. Sign up and go to Account → Tokens")
        print("3. Copy your default public token")
        print("-" * 40)
        
        api_key = input("\nPaste your MapBox token here: ").strip()
        
        if api_key:
            if update_env_file('mapbox', api_key):
                print("✅ Updated .env file with MapBox token")
                print("\n🎉 Success! Please restart Flask to use MapBox")
            else:
                print("❌ Failed to update .env file")
        else:
            print("⚠️ No token provided.")
    
    elif choice == '4':
        print("\n📍 Setting up Google Maps API")
        print("-" * 40)
        print("1. Opening Google Cloud Console...")
        webbrowser.open("https://console.cloud.google.com/google/maps-apis/")
        print("2. Create a project and enable Maps & Roads APIs")
        print("3. Create credentials (API key)")
        print("-" * 40)
        
        api_key = input("\nPaste your Google Maps API key here: ").strip()
        
        if api_key:
            if update_env_file('google', api_key):
                print("✅ Updated .env file with Google Maps API key")
                print("\n🎉 Success! Please restart Flask to use Google Maps")
            else:
                print("❌ Failed to update .env file")
        else:
            print("⚠️ No API key provided.")
    
    else:
        print("\n⚠️ Continuing with demo traffic data.")
        print("You can run this script again anytime to add real traffic data.")
    
    print("\n" + "=" * 60)
    print("Setup complete! Your app now has:")
    print("✅ Real-time weather data (OpenWeatherMap)")
    print("✅ Historical accident analysis")
    print("✅ AI risk prediction model")
    
    # Check which traffic provider is configured
    if os.environ.get('TOMTOM_API_KEY'):
        print("✅ Real traffic data (TomTom)")
    elif os.environ.get('HERE_API_KEY'):
        print("✅ Real traffic data (HERE)")
    elif os.environ.get('MAPBOX_API_KEY'):
        print("✅ Real traffic data (MapBox)")
    elif os.environ.get('GOOGLE_MAPS_API_KEY'):
        print("✅ Real traffic data (Google)")
    else:
        print("⚠️ Demo traffic data (add API key for real data)")
    
    print("=" * 60)

def test_traffic_integration():
    """Test if traffic integration is working"""
    try:
        response = requests.get('http://localhost:5000/api/traffic/current?lat=51.5074&lon=-0.1278')
        if response.status_code == 200:
            data = response.json()
            if data['traffic']['provider'] != 'demo':
                print(f"✅ Successfully fetching real traffic data from {data['traffic']['provider']}")
                print(f"   Current congestion: {data['traffic']['congestion_percentage']}%")
                print(f"   Average speed: {data['traffic']['average_speed_kmh']} km/h")
            else:
                print("⚠️ Still using demo data. Restart Flask app to activate real traffic.")
    except:
        print("⚠️ Flask app not running. Start it to test traffic integration.")

if __name__ == "__main__":
    main()
