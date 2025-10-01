#!/usr/bin/env python3
import sys
sys.path.append('.')
from app_real_api import INDIAN_CITIES, OPENWEATHER_API_KEY, TOMTOM_API_KEY
import os

print("=== VERIFICATION TEST ===")
print(f"OpenWeather API: {'LOADED' if OPENWEATHER_API_KEY else 'MISSING'}")
print(f"TomTom API: {'LOADED' if TOMTOM_API_KEY else 'MISSING'}")
print(f"Total cities: {len(INDIAN_CITIES)}")

# Test major cities
major = ['mumbai', 'delhi', 'pune', 'bangalore', 'chennai']
print("\nMajor cities:")
for city in major:
    if city in INDIAN_CITIES:
        data = INDIAN_CITIES[city]
        print(f"  {data['name']}: OK")
    else:
        print(f"  {city}: MISSING")

# Test small cities
small = ['solapur', 'satara', 'sangli', 'kolhapur', 'nashik']
print("\nSmall cities:")
for city in small:
    if city in INDIAN_CITIES:
        data = INDIAN_CITIES[city]
        print(f"  {data['name']}: OK")
    else:
        print(f"  {city}: MISSING")

# Test model
model_path = "outputs/quick_fixed/best.pt"
print(f"\nAI Model: {'FOUND' if os.path.exists(model_path) else 'MISSING'}")

print("\n=== READY TO START ===")
print("Run: python app_real_api.py")
print("Test at: http://localhost:5000")