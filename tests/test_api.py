import os
import json
import pytest

# Force demo mode to avoid live API calls during tests
os.environ['OPENWEATHER_API_KEY'] = 'demo_key'

from app import app  # noqa: E402

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'RoadSafe AI' in resp.data  # from index_new.html


def test_predict_risk_minimal_payload(client):
    payload = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "hour": 12,
        "day_of_week": 3,
        "month": 6,
        "weather_conditions": 1,
        "road_surface": 0,
        "speed_limit": 30
    }
    resp = client.post('/predict_risk', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'risk_value' in data
    assert 'risk_level' in data
    assert 'prediction_source' in data


def test_fetch_location_data_demo(client):
    # demo mode returns simulated payload with expected keys
    resp = client.post('/fetch_location_data', json={"latitude": 51.5, "longitude": -0.1})
    assert resp.status_code == 200
    data = resp.get_json()
    for k in [
        'main_weather','description','temperature','humidity','wind_speed',
        'hour','day_of_week','month','light_conditions','weather_conditions',
        'road_surface','speed_limit','road_type','junction_detail','data_source','api_status'
    ]:
        assert k in data


def test_geocode_city_demo_fallback(client):
    resp = client.post('/geocode_city', json={"city": "London"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'latitude' in data and 'longitude' in data


def test_explain_global_missing(client):
    # Likely 404 if ranking not generated
    resp = client.get('/explain/global')
    assert resp.status_code in (200, 404)


def test_explain_instance_model_state(client):
    # In demo, model may not be loaded; ensure JSON error or contribution payload is returned
    payload = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "hour": 12,
        "day_of_week": 3,
        "month": 6,
        "weather_conditions": 1,
        "road_surface": 0,
        "speed_limit": 30
    }
    resp = client.post('/explain/instance', json=payload)
    assert resp.status_code in (200, 500)
    data = resp.get_json()
    assert isinstance(data, dict)
