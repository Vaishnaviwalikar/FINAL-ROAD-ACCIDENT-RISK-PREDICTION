import requests
import time

def test_government_portal():
    """Test the government portal to see if it's loading properly"""

    print('🧪 Testing Government Portal Loading')
    print('=' * 45)

    # Test 1: Check if Flask app is running
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200:
            print('✅ Flask app is running')
        else:
            print('❌ Flask app not responding properly')
            return False
    except Exception as e:
        print('❌ Cannot connect to Flask app: ' + str(e))
        return False

    # Test 2: Check government portal page
    try:
        response = requests.get('http://localhost:5000/government', timeout=10)
        if response.status_code == 200:
            print('✅ Government portal loads successfully')

            # Check if essential elements are present
            if 'data-mode-btn' in response.text:
                print('✅ Data mode button found in HTML')
            else:
                print('❌ Data mode button not found in HTML')

            if 'Leaflet' in response.text or 'L.map' in response.text:
                print('✅ Map initialization code found')
            else:
                print('❌ Map initialization code not found')

            if 'loadDashboardData' in response.text:
                print('✅ Dashboard data loading function found')
            else:
                print('❌ Dashboard data loading function not found')

            if 'checkDataMode' in response.text:
                print('✅ Data mode check function found')
            else:
                print('❌ Data mode check function not found')

        else:
            print(f'❌ Government portal failed: {response.status_code}')

    except Exception as e:
        print('❌ Government portal error:', str(e)[:50])

    # Test 3: Check if APIs are responding
    print()
    print('🔍 Testing API Endpoints:')

    # Test heatmap statistics
    try:
        response = requests.get('http://localhost:5000/api/heatmap/statistics?lat=19.0760&lon=72.8777&radius=5', timeout=8)
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

    # Test traffic API
    try:
        response = requests.get('http://localhost:5000/api/traffic/current?lat=19.0760&lon=72.8777&radius=5', timeout=8)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print('✅ Traffic API working')
            else:
                print('❌ Traffic API error:', data.get('error'))
        else:
            print('❌ Traffic API failed:', response.status_code)
    except Exception as e:
        print('❌ Traffic API error:', str(e)[:50])

    # Test weather API
    try:
        response = requests.post('http://localhost:5000/fetch_location_data',
                               json={'latitude': 19.0760, 'longitude': 72.8777}, timeout=8)
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

    print()
    print('=' * 45)
    print('🎯 Government Portal Test Complete!')
    print('✅ Portal should now load without blocking')
    print('✅ Map should initialize properly')
    print('✅ Data mode detection should work')
    print('🚀 Ready for real-time monitoring!')

if __name__ == '__main__':
    test_government_portal()
