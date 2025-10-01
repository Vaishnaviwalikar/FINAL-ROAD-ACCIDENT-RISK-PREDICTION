"""
Traffic API Endpoints for Real-time Traffic Integration
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
import logging
from typing import Dict, Any

from .traffic_service import traffic_service, TrafficProvider

logger = logging.getLogger(__name__)
traffic_bp = Blueprint('traffic', __name__)

@traffic_bp.route('/api/traffic/current', methods=['GET'])
def get_current_traffic():
    """Get current traffic conditions for a location"""
    try:
        lat = float(request.args.get('lat', 51.5074))
        lon = float(request.args.get('lon', -0.1278))
        radius = float(request.args.get('radius', 1.0))  # km
        provider = request.args.get('provider')  # Optional specific provider
        
        # Get traffic data
        if provider:
            try:
                provider_enum = TrafficProvider[provider.upper()]
            except KeyError:
                provider_enum = None
        else:
            provider_enum = None
        
        traffic_data = traffic_service.get_traffic_data(lat, lon, radius, provider_enum)
        
        # Calculate risk factor
        risk_factor = traffic_service.calculate_traffic_risk_factor(traffic_data)
        
        return jsonify({
            'success': True,
            'location': {'lat': lat, 'lon': lon},
            'radius_km': radius,
            'traffic': {
                'congestion_level': round(traffic_data.congestion_level, 3),
                'congestion_percentage': round(traffic_data.congestion_level * 100, 1),
                'average_speed_kmh': round(traffic_data.average_speed, 1),
                'free_flow_speed_kmh': round(traffic_data.free_flow_speed, 1),
                'current_travel_time_seconds': traffic_data.current_travel_time,
                'free_flow_travel_time_seconds': traffic_data.free_flow_travel_time,
                'delay_seconds': traffic_data.current_travel_time - traffic_data.free_flow_travel_time,
                'incidents_count': len(traffic_data.incidents),
                'incidents': traffic_data.incidents[:5],  # Limit to 5 incidents
                'confidence': round(traffic_data.confidence, 2),
                'provider': traffic_data.provider,
                'timestamp': traffic_data.timestamp.isoformat()
            },
            'risk_analysis': {
                'traffic_risk_factor': round(risk_factor, 3),
                'risk_level': _get_risk_level(risk_factor),
                'contributing_factors': _get_contributing_factors(traffic_data, risk_factor)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting traffic data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@traffic_bp.route('/api/traffic/providers', methods=['GET'])
def get_available_providers():
    """Get list of available traffic data providers"""
    try:
        providers = traffic_service._get_available_providers()
        
        provider_info = []
        for provider in providers:
            info = {
                'name': provider.value,
                'display_name': provider.value.title(),
                'available': True
            }
            
            # Add provider-specific info
            if provider == TrafficProvider.TOMTOM:
                info['features'] = ['real-time flow', 'incidents', 'high accuracy']
                info['coverage'] = 'global'
            elif provider == TrafficProvider.HERE:
                info['features'] = ['real-time flow', 'predictive', 'incidents']
                info['coverage'] = 'global'
            elif provider == TrafficProvider.MAPBOX:
                info['features'] = ['routing', 'basic traffic']
                info['coverage'] = 'global'
            elif provider == TrafficProvider.GOOGLE:
                info['features'] = ['comprehensive', 'predictive']
                info['coverage'] = 'global'
            elif provider == TrafficProvider.DEMO:
                info['features'] = ['simulated data', 'always available']
                info['coverage'] = 'global'
                info['note'] = 'Demo data based on typical patterns'
            
            provider_info.append(info)
        
        return jsonify({
            'success': True,
            'providers': provider_info,
            'default_provider': providers[0].value if providers else 'demo'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@traffic_bp.route('/api/traffic/enhanced-risk', methods=['POST'])
def get_enhanced_risk():
    """Get enhanced risk prediction combining traffic with existing model"""
    try:
        data = request.get_json()
        
        lat = data.get('latitude', 51.5074)
        lon = data.get('longitude', -0.1278)
        
        # Get traffic data
        traffic_data = traffic_service.get_traffic_data(lat, lon)
        traffic_risk = traffic_service.calculate_traffic_risk_factor(traffic_data)
        
        # Get base risk from existing model (would call predict_risk here)
        base_risk = data.get('base_risk', 0.5)  # Placeholder
        
        # Combine risks (weighted average)
        traffic_weight = 0.3  # Traffic contributes 30% to final risk
        enhanced_risk = (base_risk * (1 - traffic_weight)) + (traffic_risk * traffic_weight)
        
        # Determine risk level
        if enhanced_risk < 0.33:
            risk_level = "Low"
        elif enhanced_risk < 0.66:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return jsonify({
            'success': True,
            'location': {'lat': lat, 'lon': lon},
            'risk_analysis': {
                'base_risk': round(base_risk, 3),
                'traffic_risk': round(traffic_risk, 3),
                'enhanced_risk': round(enhanced_risk, 3),
                'risk_level': risk_level,
                'traffic_impact': round((traffic_risk - base_risk) * 100, 1),  # Percentage change
                'confidence': round(traffic_data.confidence, 2)
            },
            'traffic_summary': {
                'congestion': f"{round(traffic_data.congestion_level * 100, 1)}%",
                'average_speed': f"{round(traffic_data.average_speed, 1)} km/h",
                'incidents': len(traffic_data.incidents),
                'provider': traffic_data.provider
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error calculating enhanced risk: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@traffic_bp.route('/api/traffic/route-analysis', methods=['POST'])
def analyze_route():
    """Analyze traffic along a route (list of coordinates)"""
    try:
        data = request.get_json()
        route_points = data.get('route', [])
        
        if not route_points or len(route_points) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 route points required'
            }), 400
        
        # Analyze traffic at each point
        route_analysis = []
        total_risk = 0
        
        for i, point in enumerate(route_points):
            lat = point.get('lat')
            lon = point.get('lon')
            
            if lat is None or lon is None:
                continue
            
            traffic_data = traffic_service.get_traffic_data(lat, lon, radius_km=0.5)
            risk_factor = traffic_service.calculate_traffic_risk_factor(traffic_data)
            
            route_analysis.append({
                'point_index': i,
                'location': {'lat': lat, 'lon': lon},
                'congestion': round(traffic_data.congestion_level, 3),
                'speed': round(traffic_data.average_speed, 1),
                'risk_factor': round(risk_factor, 3),
                'incidents': len(traffic_data.incidents)
            })
            
            total_risk += risk_factor
        
        # Calculate average risk for the route
        avg_risk = total_risk / len(route_analysis) if route_analysis else 0
        
        # Find highest risk segments
        high_risk_segments = [
            seg for seg in route_analysis 
            if seg['risk_factor'] > 0.6
        ]
        
        return jsonify({
            'success': True,
            'route_summary': {
                'total_points': len(route_analysis),
                'average_risk': round(avg_risk, 3),
                'risk_level': _get_risk_level(avg_risk),
                'high_risk_segments': len(high_risk_segments)
            },
            'route_analysis': route_analysis,
            'recommendations': _get_route_recommendations(avg_risk, high_risk_segments)
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing route: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _get_risk_level(risk_factor: float) -> str:
    """Convert risk factor to risk level"""
    if risk_factor < 0.33:
        return "Low"
    elif risk_factor < 0.66:
        return "Medium"
    else:
        return "High"

def _get_contributing_factors(traffic_data, risk_factor: float) -> list:
    """Get list of factors contributing to risk"""
    factors = []
    
    if traffic_data.congestion_level > 0.7:
        factors.append("Heavy congestion")
    elif traffic_data.congestion_level > 0.4:
        factors.append("Moderate congestion")
    
    if traffic_data.average_speed < 30:
        factors.append("Very slow traffic")
    elif traffic_data.average_speed < 50:
        factors.append("Slow traffic")
    
    if len(traffic_data.incidents) > 0:
        factors.append(f"{len(traffic_data.incidents)} incident(s) reported")
    
    if traffic_data.confidence < 0.7:
        factors.append("Limited data confidence")
    
    return factors

def _get_route_recommendations(avg_risk: float, high_risk_segments: list) -> list:
    """Get recommendations based on route analysis"""
    recommendations = []
    
    if avg_risk > 0.7:
        recommendations.append("Consider alternative route - very high risk")
        recommendations.append("If travel necessary, exercise extreme caution")
    elif avg_risk > 0.5:
        recommendations.append("Moderate to high risk - drive carefully")
        recommendations.append("Allow extra travel time")
    
    if len(high_risk_segments) > 0:
        recommendations.append(f"Be especially careful at {len(high_risk_segments)} high-risk segments")
    
    # Time-based recommendations
    hour = datetime.now().hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        recommendations.append("Rush hour traffic - consider traveling at different time")
    
    return recommendations
