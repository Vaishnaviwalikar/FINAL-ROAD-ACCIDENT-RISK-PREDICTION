"""
Test script to verify model loading with the updated architecture.
"""
import os
import sys
import torch
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
        logging.FileHandler('model_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the model with the updated architecture."""
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
        
        logger.info("\n" + "="*70)
        logger.info("Testing model initialization...")
        
        # Initialize model
        model = CNNBiLSTMAttn(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=0.3
        ).to(device)
        
        # Print model architecture
        logger.info("\nModel Architecture:")
        logger.info(f"Input shape: (batch, seq_len, {in_channels})")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        # Test forward pass with dummy input
        with torch.no_grad():
            batch_size = 2
            seq_len = 24  # Example sequence length
            dummy_input = torch.randn(batch_size, seq_len, in_channels).to(device)
            
            logger.info("\nTesting forward pass...")
            output = model(dummy_input)
            logger.info(f"Input shape: {tuple(dummy_input.shape)}")
            logger.info(f"Output shape: {tuple(output.shape)}")
            logger.info(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            if output.shape == (batch_size,):
                logger.info("✅ Forward pass successful!")
            else:
                logger.error(f"❌ Unexpected output shape: {output.shape}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in test_model_loading: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting model loading test...")
    success = test_model_loading()
    
    if success:
        logger.info("\n✅ Model loading test completed successfully!")
    else:
        logger.error("\n❌ Model loading test failed!")
        sys.exit(1)
