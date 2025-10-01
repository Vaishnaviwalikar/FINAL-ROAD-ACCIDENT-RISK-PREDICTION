#!/usr/bin/env python3
"""
Quick verification script to test all functionality
"""

import subprocess
import sys
import time
import requests
import json

def run_test_script(script_name, description):
    """Run a test script and return success status"""
    print(f"\n🔍 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Test failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_server_running():
    """Test if the Flask server is running"""
    print("\n🌐 Testing Server Status")
    print("=" * 50)
    
    try:
        response = requests.get("http://127.0.0.1:5000/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   API mode: {data.get('api_mode', 'unknown')}")
            print(f"   Features: {data.get('features_count', 0)}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running")
        print("   Please start the server with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Server test error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚗 Road Traffic Risk Prediction - Complete Verification")
    print("=" * 60)
    
    # Test enhanced geocoding
    geocoding_success = run_test_script(
        "scripts/enhance_geocoding.py",
        "Testing Enhanced Geocoding (Multiple Services)"
    )
    
    # Test server status
    server_success = test_server_running()
    
    # If server is running, run comprehensive tests
    if server_success:
        system_success = run_test_script(
            "scripts/test_system.py",
            "Testing Complete System (Predictions, Weather, Cities)"
        )
    else:
        system_success = False
        print("\n⚠️ Skipping system tests - server not running")
    
    # Summary
    print("\n📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Enhanced Geocoding", geocoding_success),
        ("Server Status", server_success),
        ("System Tests", system_success if server_success else None)
    ]
    
    passed = sum(1 for _, result in tests if result is True)
    total = sum(1 for _, result in tests if result is not None)
    
    for test_name, result in tests:
        if result is True:
            print(f"✅ {test_name}: PASSED")
        elif result is False:
            print(f"❌ {test_name}: FAILED")
        else:
            print(f"⏭️ {test_name}: SKIPPED")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total and total > 0:
        print("🎉 All systems are working correctly!")
        print("🌍 Your app can identify cities worldwide (including small ones)")
        print("🎯 Predictions are realistic and location-aware")
        print("🌤️ Real-time weather integration is working")
        print("\n✨ Ready for UI/UX redesign!")
    elif passed > 0:
        print("⚠️ Some systems are working, check failed tests above")
    else:
        print("❌ Major issues detected, please fix before proceeding")
    
    print("\n📖 Next Steps:")
    if not server_success:
        print("1. Start the server: python app.py")
        print("2. Re-run verification: python verify_all.py")
    else:
        print("1. Test the web interface: http://127.0.0.1:5000")
        print("2. Try different cities (small and large)")
        print("3. Verify predictions make sense")
        print("4. Proceed with UI/UX improvements")

if __name__ == "__main__":
    main()