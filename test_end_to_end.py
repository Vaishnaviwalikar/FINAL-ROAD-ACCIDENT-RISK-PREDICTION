"""
End-to-end test for the prediction pipeline.
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('e2e_test.log')
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

def test_prediction():
    """Test the prediction function with sample input."""
    try:
        # Import the prediction function and model loading
        from app_hybrid import predict_risk, load_model_and_preprocessing
        
        # Import global variables
        global a_feature_columns, a_model, a_config, a_device
        from app_hybrid import a_feature_columns, a_model, a_config, a_device
        
        import torch
        import numpy as np
        import pandas as pd
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # Load the model and preprocessing
        logger.info("Loading model and preprocessing...")
        load_model_and_preprocessing()
        
        # Get the feature columns after loading the model
        if a_feature_columns is None or len(a_feature_columns) == 0:
            # Try to get feature columns from the model's config if available
            if a_config and 'features' in a_config:
                a_feature_columns = a_config['features']
                logger.info(f"Using {len(a_feature_columns)} feature columns from config")
            else:
                logger.error("No feature columns loaded from the model!")
                if a_config:
                    logger.info(f"Available config keys: {list(a_config.keys())}")
                if hasattr(a_model, 'state_dict'):
                    logger.info("Model state dict keys:")
                    for name, param in a_model.state_dict().items():
                        logger.info(f"  {name}: {tuple(param.shape) if hasattr(param, 'shape') else param}")
                return False
            
        logger.info(f"Model loaded with {len(a_feature_columns)} features")
        
        # Create sample input with all expected features, initialized to 0
        sample_input = {feature: 0.0 for feature in a_feature_columns}
        
        # Set some realistic values for known features
        feature_mapping = {
            'hour': 18,  # 6 PM
            'Day_of_Week': 2,  # Tuesday
            'month': 5,  # May
            'Speed_limit': 100,  # km/h
            'Light_Conditions': 0,  # Daylight
            'Road_Surface_Conditions': 1,  # Wet
            'Weather_Conditions': 3,  # Rain
            'Road_Type': 1,  # Highway
            'Junction_Control': 0,  # No junction
            'Junction_Detail': 0,  # No junction
            'Number_of_Vehicles': 2,  # 2 vehicles
            'Number_of_Casualties': 1,  # 1 casualty
            'Latitude': 51.5074,  # Example latitude (London)
            'Longitude': -0.1278,  # Example longitude (London)
            'Pedestrian_Crossing-Physical_Facilities': 0,  # No crossing
            'Road_Segment_Id': 1,  # Example segment ID
            'week': 20,  # Example week of year
            '1st_Road_Class': 3,  # Example road class
            '2nd_Road_Number': 0,  # No second road
            'Accident_Severity': 2,  # Example severity
            'Police_Force': 1,  # Example police force
        }
        
        # Update sample input with known features
        for feature, value in feature_mapping.items():
            if feature in sample_input:
                sample_input[feature] = value
        
        # Ensure all values are numeric
        for k, v in sample_input.items():
            if not isinstance(v, (int, float)):
                sample_input[k] = 0.0
        
        logger.info(f"Sample input prepared with {len(sample_input)} features")
        logger.info(f"First 10 features: {list(sample_input.items())[:10]}...")
        
        # Log model architecture info if available
        if a_model is not None:
            logger.info(f"Model architecture: {a_model.__class__.__name__}")
            logger.info(f"Model device: {next(a_model.parameters()).device}")
            logger.info(f"Input features: {len(a_feature_columns)}")
            
            # Log model parameters for debugging
            total_params = sum(p.numel() for p in a_model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {total_params:,}")
            
        # Log config if available
        if a_config is not None:
            logger.info(f"Model config: {a_config.get('model', 'No model config')}")
        
        logger.info("\n" + "="*70)
        logger.info("Testing prediction with sample input:")
        for k, v in sample_input.items():
            logger.info(f"  {k}: {v}")
        
        # Make prediction
        logger.info("\nMaking prediction...")
        result = predict_risk(sample_input)
        
        logger.info("\nPrediction result:")
        for k, v in result.items():
            logger.info(f"  {k}: {v}")
        
        # Basic validation
        assert 'risk_level' in result, "Missing risk_level in result"
        assert result['risk_level'] in ['Low', 'Medium', 'High'], "Invalid risk level"
        assert 0 <= result['risk_value'] <= 1, "Risk value should be between 0 and 1"
        assert 0 <= result['confidence'] <= 100, "Confidence should be between 0 and 100"
        
        logger.info("\n✅ All tests passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        logger.info("="*70)
        logger.info("🚀 Starting roadSafe AI - End-to-End Test")
        logger.info("="*70 + "\n")
        
        # Test prediction
        logger.info("\n🔍 Testing model prediction...")
        success = test_prediction()
        
        if success:
            logger.info("\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"\n❌ Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
