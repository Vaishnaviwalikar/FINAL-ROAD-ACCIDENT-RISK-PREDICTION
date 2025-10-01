#!/usr/bin/env python3
"""
Test script to verify improved risk predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app import enhanced_analysis_prediction

def test_predictions():
    """Test predictions for different scenarios"""
    
    test_cases = [
        {
            "name": "London Rush Hour",
            "data": {
                "Latitude": 51.5074,
                "Longitude": -0.1278,
                "hour": 8,
                "Day_of_Week": 1,  # Monday
                "Weather_Conditions": 1,  # Fine
                "Speed_limit": 30,
                "Road_Surface_Conditions": 0,  # Dry
                "Light_Conditions": 1,  # Daylight
                "Urban_or_Rural_Area": 1,  # Urban
                "Number_of_Vehicles": 2
            }
        },
        {
            "name": "Small Town Night",
            "data": {
                "Latitude": 52.2053,
                "Longitude": 0.1218,  # Cambridge
                "hour": 23,
                "Day_of_Week": 6,  # Saturday
                "Weather_Conditions": 2,  # Raining
                "Speed_limit": 40,
                "Road_Surface_Conditions": 1,  # Wet
                "Light_Conditions": 4,  # Dark
                "Urban_or_Rural_Area": 1,  # Urban
                "Number_of_Vehicles": 1
            }
        },
        {
            "name": "Rural Highway",
            "data": {
                "Latitude": 53.4808,
                "Longitude": -2.2426,  # Manchester area
                "hour": 14,
                "Day_of_Week": 3,  # Wednesday
                "Weather_Conditions": 1,  # Fine
                "Speed_limit": 70,
                "Road_Surface_Conditions": 0,  # Dry
                "Light_Conditions": 1,  # Daylight
                "Urban_or_Rural_Area": 2,  # Rural
                "Number_of_Vehicles": 1
            }
        },
        {
            "name": "Snowy Conditions",
            "data": {
                "Latitude": 55.9533,
                "Longitude": -3.1883,  # Edinburgh
                "hour": 7,
                "Day_of_Week": 5,  # Friday
                "Weather_Conditions": 3,  # Snowing
                "Speed_limit": 30,
                "Road_Surface_Conditions": 2,  # Snow
                "Light_Conditions": 2,  # Twilight
                "Urban_or_Rural_Area": 1,  # Urban
                "Number_of_Vehicles": 2
            }
        },
        {
            "name": "Quiet Sunday Morning",
            "data": {
                "Latitude": 51.4545,
                "Longitude": -2.5879,  # Bristol
                "hour": 10,
                "Day_of_Week": 0,  # Sunday
                "Weather_Conditions": 1,  # Fine
                "Speed_limit": 30,
                "Road_Surface_Conditions": 0,  # Dry
                "Light_Conditions": 1,  # Daylight
                "Urban_or_Rural_Area": 1,  # Urban
                "Number_of_Vehicles": 1
            }
        }
    ]
    
    print("Testing Enhanced Risk Prediction System")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        result = enhanced_analysis_prediction(test_case['data'])
        
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Value: {result['risk_value']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Factors: Location={result['factors_considered']['location_risk']}, "
              f"Time={result['factors_considered']['time_risk']}h, "
              f"Weather={result['factors_considered']['weather_impact']}, "
              f"Speed={result['factors_considered']['speed_factor']}mph")

if __name__ == "__main__":
    test_predictions()