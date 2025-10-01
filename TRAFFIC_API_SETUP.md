# Traffic API Setup Guide

## Available Traffic Data Providers

### 1. TomTom Traffic API (Recommended)
**Best for:** Real-time traffic flow and incidents
**Coverage:** Global
**Free Tier:** 2,500 requests/day

**How to get API Key:**
1. Go to https://developer.tomtom.com/
2. Sign up for a free account
3. Create a new app in the dashboard
4. Copy your API key

### 2. HERE Traffic API
**Best for:** Predictive traffic and flow data
**Coverage:** Global
**Free Tier:** 250,000 requests/month

**How to get API Key:**
1. Go to https://developer.here.com/
2. Sign up for a free account
3. Create a new project
4. Generate REST API key

### 3. MapBox Traffic API
**Best for:** Basic traffic data with maps
**Coverage:** Global
**Free Tier:** 100,000 requests/month

**How to get API Key:**
1. Go to https://www.mapbox.com/
2. Sign up for a free account
3. Go to Account → Tokens
4. Copy your default public token or create a new one

### 4. Google Maps Platform
**Best for:** Comprehensive traffic data
**Coverage:** Global
**Free Tier:** $200 credit/month

**How to get API Key:**
1. Go to https://cloud.google.com/maps-platform/
2. Create a Google Cloud account
3. Enable Maps JavaScript API and Roads API
4. Create credentials (API key)

## Quick Setup

1. Choose one or more providers from above
2. Get your API keys (free tiers available)
3. Add to your `.env` file:

```env
# Traffic API Keys (add one or more)
TOMTOM_API_KEY=your_tomtom_key_here
HERE_API_KEY=your_here_key_here
MAPBOX_API_KEY=your_mapbox_key_here
GOOGLE_MAPS_API_KEY=your_google_key_here
```

4. Restart your Flask application
5. The system will automatically use the best available provider

## Testing Your Setup

After adding API keys, test with:

```bash
# Check available providers
curl http://localhost:5000/api/traffic/providers

# Get current traffic
curl "http://localhost:5000/api/traffic/current?lat=51.5074&lon=-0.1278"
```

## Priority Order

The system will try providers in this order:
1. TomTom (if key available)
2. HERE (if key available)
3. MapBox (if key available)
4. Google (if key available)
5. Demo data (fallback)
