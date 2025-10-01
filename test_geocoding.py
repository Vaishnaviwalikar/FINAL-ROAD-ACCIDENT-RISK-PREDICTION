#!/usr/bin/env python3
"""
Quick test script for geocoding endpoint
"""
import requests
import json

def test_geocoding():
    """Test the geocoding endpoint"""
    try:
        payload = {'city': 'Manchester'}
        response = requests.post('http://localhost:5000/geocode_city', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.text}')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Geocoding successful: {data.get('city')} at {data.get('latitude')}, {data.get('longitude')}")
        else:
            print(f"❌ Geocoding failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing geocoding: {e}")

if __name__ == "__main__":
    test_geocoding()
