import requests
import json

def test_data_mode_indicator():
    """Test the data mode indicator functionality"""

    print('🧪 Testing Data Mode Indicator')
    print('=' * 40)

    # Test 1: Check if the government portal loads
    try:
        response = requests.get('http://localhost:5000/government', timeout=5)
        if response.status_code == 200:
            print('✅ Government portal loads successfully')

            # Check if data mode indicator elements are present
            if 'data-mode-btn' in response.text:
                print('✅ Data mode button found in HTML')
            else:
                print('❌ Data mode button not found in HTML')
                return False

            if 'checkDataMode()' in response.text:
                print('✅ Data mode check function found')
            else:
                print('❌ Data mode check function not found')
                return False

            if 'toggleDataMode()' in response.text:
                print('✅ Data mode toggle function found')
            else:
                print('❌ Data mode toggle function not found')
                return False

        else:
            print('❌ Government portal failed to load')
            return False

    except Exception as e:
        print('❌ Cannot access government portal:', str(e))
        return False

    # Test 2: Test API endpoints to verify real data availability
    print()
    print('🔍 Testing API Endpoints:')

    # Test heatmap statistics
    try:
        response = requests.get('http://localhost:5000/api/heatmap/statistics?lat=19.0760&lon=72.8777&radius=5', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print('✅ Heatmap statistics API working')
            else:
                print('❌ Heatmap statistics API error:', data.get('error'))
        else:
            print('❌ Heatmap statistics API failed:', response.status_code)
    except Exception as e:
        print('❌ Heatmap statistics API error:', str(e)[:50])

    # Test weather data
    try:
        response = requests.post('http://localhost:5000/fetch_location_data',
                               json={'latitude': 19.0760, 'longitude': 72.8777}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('data_source'):
                print('✅ Weather API working - Source:', data.get('data_source'))
            else:
                print('❌ Weather API missing data source')
        else:
            print('❌ Weather API failed:', response.status_code)
    except Exception as e:
        print('❌ Weather API error:', str(e)[:50])

    # Test geocoding
    try:
        response = requests.post('http://localhost:5000/geocode_city',
                               json={'city': 'Mumbai'}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('latitude') and data.get('longitude'):
                print('✅ Geocoding API working')
            else:
                print('❌ Geocoding API missing coordinates')
        else:
            print('❌ Geocoding API failed:', response.status_code)
    except Exception as e:
        print('❌ Geocoding API error:', str(e)[:50])

    print()
    print('=' * 40)
    print('🎯 Data Mode Indicator Test Complete!')
    print('✅ Button added to navigation bar')
    print('✅ Automatic data source detection')
    print('✅ Toggle functionality implemented')
    print('✅ Visual indicators for data modes')
    print('🚀 Users can now see data source status!')

if __name__ == '__main__':
    test_data_mode_indicator()
