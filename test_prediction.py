"""
Test script to verify model prediction with sample input data.
"""
import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_prediction():
    """Test model prediction with sample input data."""
    try:
        from src.model import CNNBiLSTMAttn
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Model parameters from the checkpoint
        in_channels = 128  # Expected input features
        cnn_channels = (32, 32)  # Matches checkpoint
        lstm_hidden = 128  # Matches checkpoint
        lstm_layers = 2    # Matches checkpoint
        seq_len = 24       # Example sequence length (24 hours)
        
        logger.info("\n" + "="*70)
        logger.info("Testing model prediction with sample input...")
        
        # Initialize model
        model = CNNBiLSTMAttn(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=0.3
        ).to(device)
        model.eval()  # Set to evaluation mode
        
        # Create sample input data (normalized between 0 and 1)
        batch_size = 2
        sample_input = torch.rand(batch_size, seq_len, in_channels).to(device)
        
        logger.info(f"Sample input shape: {tuple(sample_input.shape)}")
        logger.info(f"Input range: [{sample_input.min().item():.4f}, {sample_input.max().item():.4f}]")
        
        # Make prediction
        with torch.no_grad():
            output = model(sample_input)
        
        logger.info(f"Output shape: {tuple(output.shape)}")
        logger.info(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        logger.info(f"Sample predictions: {output.tolist()}")
        
        # Verify output is in expected range (0-1 for sigmoid)
        if (output >= 0).all() and (output <= 1).all():
            logger.info("✅ Output is in expected range [0, 1]")
        else:
            logger.warning("⚠️ Output is outside expected range [0, 1]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in test_prediction: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting prediction test...")
    success = test_prediction()
    
    if success:
        logger.info("\n✅ Prediction test completed successfully!")
    else:
        logger.error("\n❌ Prediction test failed!")
        sys.exit(1)
