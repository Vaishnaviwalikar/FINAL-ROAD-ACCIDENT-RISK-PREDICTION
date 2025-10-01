#!/usr/bin/env python3
"""
Enhanced Geocoding with Multiple Fallbacks
Improves city identification for small cities worldwide
"""

import requests
import json
from typing import Dict, Optional, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class EnhancedGeocoder:
    def __init__(self):
        self.openweather_key = os.getenv('OPENWEATHER_API_KEY')
        self.tomtom_key = os.getenv('TOMTOM_API_KEY')
        
    def geocode_openweather(self, city: str) -> Optional[Dict]:
        """Primary: OpenWeatherMap Geocoding API"""
        if not self.openweather_key:
            return None
            
        try:
            url = f"http://api.openweathermap.org/geo/1.0/direct"
            params = {
                'q': city,
                'limit': 1,
                'appid': self.openweather_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        'lat': data[0]['lat'],
                        'lon': data[0]['lon'],
                        'name': data[0]['name'],
                        'country': data[0]['country'],
                        'source': 'OpenWeather'
                    }
        except Exception:
            pass
        return None
    
    def geocode_tomtom(self, city: str) -> Optional[Dict]:
        """Fallback 1: TomTom Search API"""
        if not self.tomtom_key:
            return None
            
        try:
            url = f"https://api.tomtom.com/search/2/geocode/{city}.json"
            params = {
                'key': self.tomtom_key,
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    pos = result['position']
                    return {
                        'lat': pos['lat'],
                        'lon': pos['lon'],
                        'name': result['address'].get('municipality', city),
                        'country': result['address'].get('country', ''),
                        'source': 'TomTom'
                    }
        except Exception:
            pass
        return None
    
    def geocode_nominatim(self, city: str) -> Optional[Dict]:
        """Fallback 2: OpenStreetMap Nominatim (Free)"""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': city,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'RoadRiskPredictor/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    return {
                        'lat': float(result['lat']),
                        'lon': float(result['lon']),
                        'name': result.get('display_name', city).split(',')[0],
                        'country': result.get('address', {}).get('country', ''),
                        'source': 'Nominatim'
                    }
        except Exception:
            pass
        return None
    
    def geocode_with_fallbacks(self, city: str) -> Tuple[bool, Dict]:
        """Try multiple geocoding services with fallbacks"""
        
        # Clean city input
        city = city.strip()
        if not city:
            return False, {"error": "Empty city name"}
        
        # Try primary service
        result = self.geocode_openweather(city)
        if result:
            return True, {
                "success": True,
                "latitude": result['lat'],
                "longitude": result['lon'],
                "city_name": result['name'],
                "country": result['country'],
                "source": result['source']
            }
        
        # Try TomTom fallback
        result = self.geocode_tomtom(city)
        if result:
            return True, {
                "success": True,
                "latitude": result['lat'],
                "longitude": result['lon'],
                "city_name": result['name'],
                "country": result['country'],
                "source": result['source']
            }
        
        # Try Nominatim fallback
        result = self.geocode_nominatim(city)
        if result:
            return True, {
                "success": True,
                "latitude": result['lat'],
                "longitude": result['lon'],
                "city_name": result['name'],
                "country": result['country'],
                "source": result['source']
            }
        
        # All services failed
        return False, {
            "success": False,
            "error": f"City '{city}' not found in any geocoding service",
            "suggestions": [
                "Check spelling",
                "Try with country name (e.g., 'Paris, France')",
                "Use full city name instead of abbreviations"
            ]
        }

def test_enhanced_geocoding():
    """Test enhanced geocoding with challenging cities"""
    geocoder = EnhancedGeocoder()
    
    # Test cities including very small ones
    test_cities = [
        "Dharamshala, India",
        "Shimla, India", 
        "Manali, India",
        "Rishikesh, India",
        "Haridwar, India",
        "Nainital, India",
        "Mussoorie, India",
        "Dehradun, India",
        "Badrinath, India",
        "Kedarnath, India",
        "Bath, UK",
        "Canterbury, UK",
        "Salem, Oregon, US",
        "Bend, Oregon, US",
        "Nara, Japan",
        "Kamakura, Japan"
    ]
    
    print("🌍 Testing Enhanced Geocoding...")
    print("=" * 50)
    
    success_count = 0
    for city in test_cities:
        success, result = geocoder.geocode_with_fallbacks(city)
        if success:
            success_count += 1
            print(f"✅ {city}: ({result['latitude']:.4f}, {result['longitude']:.4f}) via {result['source']}")
        else:
            print(f"❌ {city}: {result['error']}")
    
    print(f"\n📊 Success Rate: {success_count}/{len(test_cities)} ({success_count/len(test_cities)*100:.1f}%)")

if __name__ == "__main__":
    test_enhanced_geocoding()