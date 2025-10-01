"""
Historical Data Models for Road Accident Risk Prediction
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
from typing import Optional, List, Dict, Any

Base = declarative_base()

class HistoricalAccident(Base):
    """Historical accident data model"""
    __tablename__ = 'historical_accidents'

    id = Column(Integer, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    accident_date = Column(DateTime, nullable=False)
    severity = Column(Integer)  # 1-4 scale
    casualties = Column(Integer, default=0)
    vehicles_involved = Column(Integer, default=1)
    road_type = Column(String(50))
    junction_detail = Column(Integer)
    weather_conditions = Column(String(50))
    road_surface = Column(String(50))
    light_conditions = Column(Integer)
    speed_limit = Column(Integer)
    urban_rural = Column(String(20))
    police_force = Column(String(100))
    local_authority = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class HistoricalWeather(Base):
    """Historical weather data model"""
    __tablename__ = 'historical_weather'

    id = Column(Integer, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    weather_date = Column(DateTime, nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    precipitation = Column(Float)
    wind_speed = Column(Float)
    weather_main = Column(String(50))
    weather_description = Column(String(100))
    visibility = Column(Float)
    pressure = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class HistoricalTraffic(Base):
    """Historical traffic data model"""
    __tablename__ = 'historical_traffic'

    id = Column(Integer, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    traffic_date = Column(DateTime, nullable=False)
    congestion_level = Column(Float)  # 0-1 scale
    average_speed = Column(Float)
    traffic_flow = Column(Integer)  # vehicles per hour
    incident_count = Column(Integer, default=0)
    road_closure = Column(Boolean, default=False)
    source = Column(String(50))  # 'google_maps', 'tomtom', etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RiskAnalysis(Base):
    """Pre-computed risk analysis results"""
    __tablename__ = 'risk_analysis'

    id = Column(Integer, primary_key=True)
    analysis_date = Column(DateTime, nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    analysis_type = Column(String(50))  # 'daily', 'weekly', 'monthly', 'seasonal'
    risk_score = Column(Float)
    accident_count = Column(Integer)
    weather_factor = Column(Float)
    traffic_factor = Column(Float)
    peak_hours = Column(Text)  # JSON string of peak hours
    trends_data = Column(Text)  # JSON string of trend data
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class GovernmentReport(Base):
    """Government and authority reports"""
    __tablename__ = 'government_reports'

    id = Column(Integer, primary_key=True)
    report_date = Column(DateTime, nullable=False)
    authority_name = Column(String(100))
    report_type = Column(String(50))  # 'policy_impact', 'resource_allocation', 'safety_report'
    location = Column(String(200))
    content = Column(Text)
    recommendations = Column(Text)
    impact_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Database setup
DATABASE_URL = "sqlite:///historical_data.db"  # Can be changed to PostgreSQL for production
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_tables()
    print("✅ Historical data database tables created successfully!")
