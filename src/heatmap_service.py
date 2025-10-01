"""
Interactive Risk Heatmap Service
Generates risk heatmaps for visual representation of accident risk across geographic areas
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import math

from .traffic_service import traffic_service
from .historical_service import HistoricalDataService

logger = logging.getLogger(__name__)

@dataclass
class HeatmapPoint:
    """Represents a single point in the heatmap"""
    lat: float
    lon: float
    risk_level: float  # 0-1 scale
    risk_category: str  # 'low', 'medium', 'high', 'extreme'
    contributing_factors: List[str]
    confidence: float

@dataclass
class HeatmapData:
    """Complete heatmap data for a geographic area"""
    center_lat: float
    center_lon: float
    radius_km: float
    grid_resolution: int  # Number of points per side
    points: List[HeatmapPoint]
    generated_at: datetime
    parameters: Dict[str, Any]

class RiskHeatmapService:
    """Service for generating interactive risk heatmaps"""

    def __init__(self):
        self.historical_service = HistoricalDataService()
        self.grid_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def generate_heatmap(self,
                        center_lat: float,
                        center_lon: float,
                        radius_km: float = 5.0,
                        grid_resolution: int = 20,
                        time_params: Optional[Dict] = None,
                        include_traffic: bool = True,
                        include_historical: bool = True) -> HeatmapData:
        """
        Generate a risk heatmap for the specified area

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in kilometers
            grid_resolution: Number of grid points per side (20x20 = 400 points)
            time_params: Time parameters for analysis
            include_traffic: Whether to include real-time traffic data
            include_historical: Whether to include historical accident data
        """
        cache_key = self._get_cache_key(center_lat, center_lon, radius_km, grid_resolution, time_params)
        cached_data = self._get_cached_heatmap(cache_key)
        if cached_data:
            return cached_data

        # Generate grid points
        grid_points = self._generate_grid_points(center_lat, center_lon, radius_km, grid_resolution)

        # Calculate risk for each point
        heatmap_points = []
        for lat, lon in grid_points:
            risk_data = self._calculate_point_risk(
                lat, lon, time_params, include_traffic, include_historical
            )
            heatmap_points.append(risk_data)

        # Create heatmap data
        heatmap_data = HeatmapData(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            grid_resolution=grid_resolution,
            points=heatmap_points,
            generated_at=datetime.now(),
            parameters={
                'time_params': time_params,
                'include_traffic': include_traffic,
                'include_historical': include_historical
            }
        )

        # Cache the result
        self._cache_heatmap(cache_key, heatmap_data)

        return heatmap_data

    def _generate_grid_points(self, center_lat: float, center_lon: float,
                            radius_km: float, grid_resolution: int) -> List[Tuple[float, float]]:
        """Generate evenly spaced grid points within the specified radius"""
        # Calculate the angular spacing
        lat_spacing = (radius_km * 2) / (grid_resolution * 111.0)  # Approximate km per degree
        lon_spacing = (radius_km * 2) / (grid_resolution * 111.0 * abs(math.cos(math.radians(center_lat))))

        points = []
        half_grid = grid_resolution // 2

        for i in range(grid_resolution):
            for j in range(grid_resolution):
                lat = center_lat + (i - half_grid) * lat_spacing
                lon = center_lon + (j - half_grid) * lon_spacing

                # Check if point is within radius
                distance = self._calculate_distance(center_lat, center_lon, lat, lon)
                if distance <= radius_km:
                    points.append((lat, lon))

        return points

    def _calculate_point_risk(self,
                            lat: float,
                            lon: float,
                            time_params: Optional[Dict],
                            include_traffic: bool,
                            include_historical: bool) -> HeatmapPoint:
        """Calculate risk level for a specific point"""

        # Default time parameters
        if time_params is None:
            time_params = {
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday() + 1,
                'month': datetime.now().month
            }

        total_risk = 0.0
        contributing_factors = []
        confidence_factors = []

        # 1. Historical Risk (40% weight)
        if include_historical:
            historical_risk = self._get_historical_risk(lat, lon, radius_km=1.0)
            total_risk += historical_risk * 0.4
            if historical_risk > 0.3:
                contributing_factors.append("High historical accident rate")
            confidence_factors.append(0.8)  # Historical data is reliable

        # 2. Traffic Risk (35% weight)
        if include_traffic:
            try:
                traffic_data = traffic_service.get_traffic_data(lat, lon, radius_km=0.5)
                traffic_risk = traffic_service.calculate_traffic_risk_factor(traffic_data)
                total_risk += traffic_risk * 0.35
                if traffic_risk > 0.4:
                    contributing_factors.append("Heavy traffic congestion")
                confidence_factors.append(traffic_data.confidence)
            except Exception as e:
                logger.warning(f"Failed to get traffic data for {lat},{lon}: {e}")
                confidence_factors.append(0.5)

        # 3. Time-based Risk (15% weight)
        time_risk = self._calculate_time_risk(time_params)
        total_risk += time_risk * 0.15
        if time_risk > 0.5:
            contributing_factors.append("Peak traffic hours")
        confidence_factors.append(0.9)

        # 4. Location Risk (10% weight)
        location_risk = self._calculate_location_risk(lat, lon)
        total_risk += location_risk * 0.1
        if location_risk > 0.6:
            contributing_factors.append("High-risk road type")
        confidence_factors.append(0.7)

        # Calculate average confidence
        avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

        # Adjust risk by confidence
        total_risk *= avg_confidence

        # Ensure risk is within bounds
        total_risk = max(0.0, min(1.0, total_risk))

        # Determine risk category
        if total_risk < 0.25:
            risk_category = 'low'
        elif total_risk < 0.5:
            risk_category = 'medium'
        elif total_risk < 0.75:
            risk_category = 'high'
        else:
            risk_category = 'extreme'

        return HeatmapPoint(
            lat=lat,
            lon=lon,
            risk_level=total_risk,
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            confidence=avg_confidence
        )

    def _get_historical_risk(self, lat: float, lon: float, radius_km: float) -> float:
        """Get historical risk based on accident data"""
        try:
            # Query historical data for this area
            stats = self.historical_service.get_location_statistics(
                lat=lat, lon=lon, radius_km=radius_km, days_back=365
            )

            # Calculate risk based on accident frequency
            total_accidents = stats.get('total_accidents', 0)
            avg_severity = stats.get('average_severity', 1.0)

            # Normalize to 0-1 scale
            frequency_risk = min(total_accidents / 50.0, 1.0)  # Cap at 50 accidents
            severity_risk = (avg_severity - 1) / 2.0  # Severity 1-3 -> 0-1

            return (frequency_risk * 0.7) + (severity_risk * 0.3)

        except Exception as e:
            logger.warning(f"Failed to get historical risk: {e}")
            return 0.3  # Default moderate risk

    def _calculate_time_risk(self, time_params: Dict) -> float:
        """Calculate risk based on time of day/week"""
        hour = time_params.get('hour', 12)
        day_of_week = time_params.get('day_of_week', 1)
        month = time_params.get('month', 6)

        # Peak hours (rush hour)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            hour_risk = 0.8
        elif 6 <= hour <= 22:
            hour_risk = 0.5
        else:
            hour_risk = 0.3

        # Weekend vs weekday
        if day_of_week in [6, 7]:  # Weekend
            weekend_risk = 0.6
        else:
            weekend_risk = 0.4

        # Seasonal variation (winter higher risk)
        if month in [12, 1, 2, 3]:  # Winter
            seasonal_risk = 0.7
        elif month in [6, 7, 8]:  # Summer
            seasonal_risk = 0.4
        else:
            seasonal_risk = 0.5

        return (hour_risk * 0.5) + (weekend_risk * 0.3) + (seasonal_risk * 0.2)

    def _calculate_location_risk(self, lat: float, lon: float) -> float:
        """Calculate risk based on location characteristics"""
        # Urban areas have higher risk
        if self._is_urban_area(lat, lon):
            urban_risk = 0.7
        else:
            urban_risk = 0.4

        # Road type risk (assuming major roads have higher risk)
        road_risk = 0.6  # Default moderate-high

        return (urban_risk * 0.6) + (road_risk * 0.4)

    def _is_urban_area(self, lat: float, lon: float) -> bool:
        """Determine if location is in urban area (simplified)"""
        # Simple heuristic: major cities have higher density
        urban_centers = [
            (51.5074, -0.1278),  # London
            (40.7128, -74.0060), # New York
            (19.0760, 72.8777),  # Mumbai
            (52.5200, 13.4050),  # Berlin
            (48.8566, 2.3522),   # Paris
        ]

        for center_lat, center_lon in urban_centers:
            distance = self._calculate_distance(lat, lon, center_lat, center_lon)
            if distance < 50:  # Within 50km of major city
                return True

        return False

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371.0  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _get_cache_key(self, lat: float, lon: float, radius: float, resolution: int, time_params: Optional[Dict]) -> str:
        """Generate cache key for heatmap"""
        import hashlib

        time_str = str(sorted(time_params.items())) if time_params else "default"
        key = f"heatmap_{lat}_{lon}_{radius}_{resolution}_{time_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cached_heatmap(self, cache_key: str) -> Optional[HeatmapData]:
        """Get cached heatmap data"""
        if cache_key in self.grid_cache:
            cached = self.grid_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                return cached['data']
        return None

    def _cache_heatmap(self, cache_key: str, data: HeatmapData):
        """Cache heatmap data"""
        self.grid_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }

        # Clean old cache entries
        self._clean_cache()

    def _clean_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = []

        for key, value in self.grid_cache.items():
            if (now - value['timestamp']).seconds > self.cache_ttl * 2:
                expired_keys.append(key)

        for key in expired_keys:
            del self.grid_cache[key]

# Singleton instance
heatmap_service = RiskHeatmapService()
