"""
Simple script to test loading a PyTorch model
"""
import os
import torch
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_simple_model(checkpoint_path):
    """Load a PyTorch model from checkpoint."""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 Attempting to load model from: {checkpoint_path}")
        logger.info(f"File exists: {os.path.exists(checkpoint_path)}")
        
        # Try loading with map_location='cpu' for compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info("✅ Checkpoint loaded successfully!")
        
        # Print checkpoint structure
        logger.info("\n📋 Checkpoint structure:")
        for key in checkpoint.keys():
            if hasattr(checkpoint[key], 'shape'):
                logger.info(f"  - {key}: Tensor with shape {checkpoint[key].shape}")
            elif isinstance(checkpoint[key], dict):
                logger.info(f"  - {key}: Dictionary with keys {list(checkpoint[key].keys())}")
            else:
                logger.info(f"  - {key}: {type(checkpoint[key])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Try loading different model checkpoints
    checkpoints = [
        "outputs/final_fixed/best.pt",
        "outputs/uk_full_cpu/best.pt",
        "outputs/uk_top15_e50/best.pt"
    ]
    
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing: {ckpt}")
            success = load_simple_model(ckpt)
            if success:
                logger.info("✅ Successfully loaded the model!")
                break
        else:
            logger.warning(f"Checkpoint not found: {ckpt}")
