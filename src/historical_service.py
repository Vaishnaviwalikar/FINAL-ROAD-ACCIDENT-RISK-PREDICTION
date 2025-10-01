"""
Historical Data Service for Road Accident Risk Prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
import calendar

from .historical_models import (
    HistoricalAccident, HistoricalWeather, HistoricalTraffic,
    RiskAnalysis, GovernmentReport, get_db, SessionLocal
)

logger = logging.getLogger(__name__)

class HistoricalDataService:
    """Service for managing historical accident, weather, and traffic data"""

    def __init__(self):
        self.db_session = SessionLocal()

    def ingest_accident_data(self, accidents_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest historical accident data"""
        try:
            added_count = 0
            for accident_data in accidents_data:
                accident = HistoricalAccident(
                    latitude=accident_data.get('latitude'),
                    longitude=accident_data.get('longitude'),
                    accident_date=accident_data.get('accident_date'),
                    severity=accident_data.get('severity', 1),
                    casualties=accident_data.get('casualties', 0),
                    vehicles_involved=accident_data.get('vehicles_involved', 1),
                    road_type=accident_data.get('road_type'),
                    junction_detail=accident_data.get('junction_detail'),
                    weather_conditions=accident_data.get('weather_conditions'),
                    road_surface=accident_data.get('road_surface'),
                    light_conditions=accident_data.get('light_conditions'),
                    speed_limit=accident_data.get('speed_limit'),
                    urban_rural=accident_data.get('urban_rural'),
                    police_force=accident_data.get('police_force'),
                    local_authority=accident_data.get('local_authority'),
                    description=accident_data.get('description')
                )
                self.db_session.add(accident)
                added_count += 1

            self.db_session.commit()
            return {
                "success": True,
                "message": f"Successfully ingested {added_count} accident records",
                "records_added": added_count
            }
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error ingesting accident data: {e}")
            return {"success": False, "error": str(e)}

    def ingest_weather_data(self, weather_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest historical weather data"""
        try:
            added_count = 0
            for weather in weather_data:
                weather_record = HistoricalWeather(
                    latitude=weather.get('latitude'),
                    longitude=weather.get('longitude'),
                    weather_date=weather.get('weather_date'),
                    temperature=weather.get('temperature'),
                    humidity=weather.get('humidity'),
                    precipitation=weather.get('precipitation'),
                    wind_speed=weather.get('wind_speed'),
                    weather_main=weather.get('weather_main'),
                    weather_description=weather.get('weather_description'),
                    visibility=weather.get('visibility'),
                    pressure=weather.get('pressure')
                )
                self.db_session.add(weather_record)
                added_count += 1

            self.db_session.commit()
            return {
                "success": True,
                "message": f"Successfully ingested {added_count} weather records",
                "records_added": added_count
            }
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error ingesting weather data: {e}")
            return {"success": False, "error": str(e)}

    def get_accident_statistics(self, start_date: datetime, end_date: datetime,
                              lat: float, lon: float, radius_km: float = 5.0) -> Dict[str, Any]:
        """Get accident statistics for a given area and time period"""
        try:
            # Calculate bounding box for the radius
            lat_range = radius_km / 111.0  # Approximate degrees per km
            lon_range = radius_km / (111.0 * np.cos(np.radians(lat)))

            accidents = self.db_session.query(HistoricalAccident).filter(
                and_(
                    HistoricalAccident.accident_date >= start_date,
                    HistoricalAccident.accident_date <= end_date,
                    HistoricalAccident.latitude.between(lat - lat_range, lat + lat_range),
                    HistoricalAccident.longitude.between(lon - lon_range, lon + lon_range)
                )
            ).all()

            if not accidents:
                return {"success": True, "data": [], "total_count": 0}

            # Calculate statistics
            total_accidents = len(accidents)
            severity_counts = {}
            weather_counts = {}
            road_type_counts = {}
            hourly_distribution = {i: 0 for i in range(24)}
            daily_distribution = {i: 0 for i in range(7)}  # 0=Monday, 6=Sunday

            for accident in accidents:
                # Severity statistics
                severity = accident.severity or 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

                # Weather statistics
                weather = accident.weather_conditions or 'Unknown'
                weather_counts[weather] = weather_counts.get(weather, 0) + 1

                # Road type statistics
                road_type = accident.road_type or 'Unknown'
                road_type_counts[road_type] = road_type_counts.get(road_type, 0) + 1

                # Hourly distribution
                hour = accident.accident_date.hour
                hourly_distribution[hour] += 1

                # Daily distribution
                day = accident.accident_date.weekday()
                daily_distribution[day] += 1

            # Calculate peak hours
            peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:5]

            # Calculate risk score (accidents per day)
            days_diff = (end_date - start_date).days or 1
            risk_score = total_accidents / days_diff

            return {
                "success": True,
                "total_accidents": total_accidents,
                "risk_score": round(risk_score, 3),
                "severity_distribution": severity_counts,
                "weather_distribution": weather_counts,
                "road_type_distribution": road_type_counts,
                "hourly_distribution": hourly_distribution,
                "daily_distribution": daily_distribution,
                "peak_hours": peak_hours,
                "time_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days_diff
                }
            }

        except Exception as e:
            logger.error(f"Error getting accident statistics: {e}")
            return {"success": False, "error": str(e)}

    def identify_peak_risk_hours(self, lat: float, lon: float, radius_km: float = 5.0,
                                days_back: int = 365) -> Dict[str, Any]:
        """Identify peak risk hours based on historical data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            stats = self.get_accident_statistics(start_date, end_date, lat, lon, radius_km)

            if not stats["success"]:
                return stats

            # Find hours with highest accident frequency
            hourly_data = stats["hourly_distribution"]
            peak_hours = sorted(hourly_data.items(), key=lambda x: x[1], reverse=True)

            # Calculate risk levels for each hour
            max_accidents = max(hourly_data.values()) if hourly_data else 1
            risk_levels = {}
            for hour, count in hourly_data.items():
                risk_level = (count / max_accidents) * 100  # Percentage of peak risk
                risk_levels[hour] = {
                    "accident_count": count,
                    "risk_percentage": round(risk_level, 1),
                    "risk_category": self._categorize_risk(risk_level)
                }

            return {
                "success": True,
                "peak_hours": peak_hours[:10],  # Top 10 peak hours
                "risk_levels": risk_levels,
                "analysis_period_days": days_back,
                "total_accidents_analyzed": stats["total_accidents"]
            }

        except Exception as e:
            logger.error(f"Error identifying peak risk hours: {e}")
            return {"success": False, "error": str(e)}

    def analyze_seasonal_patterns(self, lat: float, lon: float, radius_km: float = 5.0,
                                years_back: int = 3) -> Dict[str, Any]:
        """Analyze seasonal patterns in accident data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)

            accidents = self.db_session.query(HistoricalAccident).filter(
                and_(
                    HistoricalAccident.accident_date >= start_date,
                    HistoricalAccident.accident_date <= end_date,
                    HistoricalAccident.latitude.between(lat - 0.1, lat + 0.1),
                    HistoricalAccident.longitude.between(lon - 0.1, lon + 0.1)
                )
            ).all()

            if not accidents:
                return {"success": True, "data": {}, "message": "No data available"}

            # Group by month
            monthly_data = {i: {"count": 0, "severity_sum": 0, "weather_conditions": {}}
                          for i in range(1, 13)}

            for accident in accidents:
                month = accident.accident_date.month
                monthly_data[month]["count"] += 1
                monthly_data[month]["severity_sum"] += accident.severity or 1

                # Weather condition tracking
                weather = accident.weather_conditions or 'Unknown'
                if weather not in monthly_data[month]["weather_conditions"]:
                    monthly_data[month]["weather_conditions"][weather] = 0
                monthly_data[month]["weather_conditions"][weather] += 1

            # Calculate seasonal statistics
            seasonal_stats = {}
            seasons = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Autumn': [9, 10, 11]
            }

            for season, months in seasons.items():
                season_accidents = sum(monthly_data[m]["count"] for m in months if m in monthly_data)
                avg_severity = sum(monthly_data[m]["severity_sum"] for m in months if m in monthly_data) / max(season_accidents, 1)

                seasonal_stats[season] = {
                    "total_accidents": season_accidents,
                    "average_severity": round(avg_severity, 2),
                    "monthly_breakdown": {m: monthly_data.get(m, {}) for m in months}
                }

            return {
                "success": True,
                "seasonal_statistics": seasonal_stats,
                "monthly_data": monthly_data,
                "analysis_period_years": years_back,
                "total_accidents": len(accidents)
            }

        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            return {"success": False, "error": str(e)}

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk level based on percentage"""
        if risk_percentage >= 80:
            return "Very High"
        elif risk_percentage >= 60:
            return "High"
        elif risk_percentage >= 40:
            return "Moderate"
        elif risk_percentage >= 20:
            return "Low"
        else:
            return "Very Low"

    def close(self):
        """Close database session"""
        self.db_session.close()

# Utility functions for data ingestion
def load_sample_accident_data() -> List[Dict[str, Any]]:
    """Load sample accident data for testing"""
    # This would typically load from CSV files or APIs
    # For now, return sample data structure
    return [
        {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "accident_date": datetime.now() - timedelta(days=30),
            "severity": 2,
            "casualties": 1,
            "vehicles_involved": 2,
            "road_type": "Urban",
            "junction_detail": 1,
            "weather_conditions": "Rain",
            "road_surface": "Wet",
            "light_conditions": 2,
            "speed_limit": 30,
            "urban_rural": "Urban",
            "police_force": "Metropolitan Police",
            "local_authority": "London"
        }
    ]

def load_sample_weather_data() -> List[Dict[str, Any]]:
    """Load sample weather data for testing"""
    return [
        {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "weather_date": datetime.now() - timedelta(days=30),
            "temperature": 15.5,
            "humidity": 75.0,
            "precipitation": 2.5,
            "wind_speed": 12.0,
            "weather_main": "Rain",
            "weather_description": "Light rain",
            "visibility": 8.0,
            "pressure": 1013.0
        }
    ]
