"""
Simplified model loading test script.
"""
import os
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Path to the model checkpoint
    checkpoint_path = "outputs/uk_full_cpu/best.pt"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info("Checkpoint loaded successfully")
        
        # Print checkpoint structure
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model' not in checkpoint:
            logger.error("No 'model' key in checkpoint")
            return
        
        # Print model keys
        model_keys = list(checkpoint['model'].keys())
        logger.info(f"Model has {len(model_keys)} parameters")
        logger.info(f"First 10 parameters: {model_keys[:10]}")
        
        # Import model class
        from src.model import CNNBiLSTMAttn
        
        # Create a model with default parameters
        logger.info("Creating model instance...")
        model = CNNBiLSTMAttn(
            in_channels=len(checkpoint.get('features', [])),
            cnn_channels=(64, 128),
            kernel_sizes=(3, 3),
            pool_size=2,
            fc_dim=128,
            attn_spatial_dim=64,
            attn_temporal_dim=64,
            lstm_hidden=64,
            lstm_layers=1,
            dropout=0.3
        )
        
        # Try to load the state dict
        logger.info("Loading model weights...")
        model.load_state_dict(checkpoint['model'], strict=False)
        logger.info("✅ Model loaded successfully with strict=False")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
