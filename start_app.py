#!/usr/bin/env python3
"""
Startup script for RoadSafe AI application.
"""
import os
import sys
import time

def main():
    print("🚀 Starting RoadSafe AI Application")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app_hybrid.py"):
        print("❌ Error: app_hybrid.py not found!")
        print("Please run this script from the road directory.")
        sys.exit(1)
    
    # Check if model exists
    model_path = os.path.join("outputs", "quick_fixed", "best.pt")
    if os.path.exists(model_path):
        print(f"✅ AI model found: {model_path}")
        print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print("⚠️  AI model not found - will use fallback mode")
    
    # Check environment
    api_key = os.environ.get('OPENWEATHER_API_KEY', 'demo_key')
    if api_key == 'demo_key':
        print("⚠️  Using demo weather data (set OPENWEATHER_API_KEY for live data)")
    else:
        print("✅ OpenWeatherMap API key configured")
    
    print("\n🌐 Starting Flask web server...")
    print("📍 URL: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Import and run the app
    try:
        import app_hybrid
        app_hybrid.app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()