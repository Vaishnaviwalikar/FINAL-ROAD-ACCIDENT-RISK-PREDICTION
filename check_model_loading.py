"""
Script to check model loading and feature columns.
"""
import os
import sys
import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_check.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_model_loading():
    """Check if the model loads correctly and print debug info."""
    try:
        # Import the app_hybrid module
        import app_hybrid
        
        logger.info("Checking model loading...")
        
        # Load the model
        app_hybrid.load_model_and_preprocessing()
        
        # Check if model loaded
        if app_hybrid.a_model is None:
            logger.error("[ERROR] Model failed to load!")
            return False
            
        logger.info("[SUCCESS] Model loaded successfully!")
        
        # Print model info
        logger.info(f"\nModel architecture: {type(app_hybrid.a_model).__name__}")
        logger.info(f"Model device: {next(app_hybrid.a_model.parameters()).device}")
        logger.info(f"Model in eval mode: {not app_hybrid.a_model.training}")
        
        # Print feature columns
        if app_hybrid.a_feature_columns:
            logger.info(f"\nFeature columns ({len(app_hybrid.a_feature_columns)}):")
            for i, feat in enumerate(app_hybrid.a_feature_columns[:20]):  # Print first 20 features
                logger.info(f"  {i+1}. {feat}")
            if len(app_hybrid.a_feature_columns) > 20:
                logger.info(f"  ... and {len(app_hybrid.a_feature_columns) - 20} more")
        else:
            logger.warning("No feature columns loaded!")
        
        # Print config info
        if app_hybrid.a_config:
            logger.info("\nModel config keys:")
            for k in app_hybrid.a_config.keys():
                logger.info(f"  - {k}")
        
        # Test a simple prediction
        logger.info("\nTesting prediction functionality...")
        test_input = {
            'Latitude': 51.5074,
            'Longitude': -0.1278,
            'hour': 14,
            'Day_of_Week': 3,
            'Weather_Conditions': 1,
            'Road_Surface_Conditions': 0,
            'Speed_limit': 30
        }
        
        prediction = app_hybrid.predict_risk(test_input)
        logger.info(f"Test prediction result: {prediction}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking model loading: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting model loading check...")
    success = check_model_loading()
    
    if success:
        logger.info("\n[SUCCESS] Model loading check completed successfully!")
    else:
        logger.error("\n[ERROR] Model loading check failed!")
        sys.exit(1)
