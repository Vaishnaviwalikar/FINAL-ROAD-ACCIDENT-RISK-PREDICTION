#!/usr/bin/env python3
"""
Simple script to help you set up your OpenWeatherMap API key
"""
import os

def setup_api_key():
    print("🌤️  OpenWeatherMap API Key Setup")
    print("=" * 40)
    print("Get your free API key from: https://openweathermap.org/api")
    print()
    
    api_key = input("Enter your OpenWeatherMap API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Using demo mode.")
        return
    
    if api_key == "demo_key":
        print("🧪 Using demo mode with simulated data.")
        return
    
    # Update the config_api.py file
    try:
        with open('config_api.py', 'r') as f:
            content = f.read()
        
        # Find and replace the hardcoded API key section
        lines = content.split('\n')
        new_lines = []
        in_hardcoded_section = False
        
        for line in lines:
            if '# If you want to hardcode your API key temporarily, uncomment and replace below:' in line:
                in_hardcoded_section = True
                new_lines.append(line)
                new_lines.append(f'    HARDCODED_API_KEY = "{api_key}"')
                new_lines.append('    if HARDCODED_API_KEY and HARDCODED_API_KEY != "your_actual_api_key_here":')
                new_lines.append('        return HARDCODED_API_KEY')
                continue
            elif in_hardcoded_section and line.strip().startswith('#') and 'HARDCODED_API_KEY' in line:
                continue  # Skip the commented lines
            else:
                if in_hardcoded_section and not line.strip().startswith('#') and 'HARDCODED_API_KEY' not in line:
                    in_hardcoded_section = False
                new_lines.append(line)
        
        with open('config_api.py', 'w') as f:
            f.write('\n'.join(new_lines))
        
        print(f"✅ API key configured successfully!")
        print(f"🚀 You can now restart your Flask app to use real weather data.")
        
    except Exception as e:
        print(f"❌ Error updating config file: {e}")
        print(f"💡 You can manually edit config_api.py and uncomment the HARDCODED_API_KEY line")

if __name__ == "__main__":
    setup_api_key()
