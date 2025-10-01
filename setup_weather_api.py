#!/usr/bin/env python3
"""
Setup OpenWeatherMap API key for RoadSafe AI
"""
import os

def setup_api_key():
    print("🌤️ OpenWeatherMap API Setup for RoadSafe AI")
    print("=" * 50)
    print("To get real weather data for Indian cities, you need an API key.")
    print("1. Go to: https://openweathermap.org/api")
    print("2. Sign up for free account")
    print("3. Get your API key")
    print("4. Enter it below")
    print()
    
    api_key = input("Enter your OpenWeatherMap API key (or press Enter to use demo mode): ").strip()
    
    if api_key:
        # Create .env file
        with open('.env', 'w') as f:
            f.write(f"OPENWEATHER_API_KEY={api_key}\n")
        
        # Set environment variable for current session
        os.environ['OPENWEATHER_API_KEY'] = api_key
        
        print("✅ API key saved to .env file")
        print("✅ Real weather data will be used for predictions")
    else:
        print("⚠️ Using demo mode with simulated weather data")
        print("💡 You can add API key later by running this script again")
    
    print("\n🚀 Ready to start RoadSafe AI!")
    print("Run: python run_fixed_app.py")

if __name__ == "__main__":
    setup_api_key()