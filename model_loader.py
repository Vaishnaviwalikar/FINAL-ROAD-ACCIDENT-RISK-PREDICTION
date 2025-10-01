"""
Enhanced model loading utility for RoadSafe AI
"""
import os
import sys
import logging
import torch
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler('logs/model_loader.log', maxBytes=10485760, backupCount=5)
        ]
    )
    return logging.getLogger(__name__)

def load_model():
    """Load the trained model with enhanced error handling."""
    logger = setup_logging()
    logger.info("\n" + "="*70)
    logger.info("🚀 Starting Model Loading Process")
    logger.info("="*70)
    
    # Check PyTorch installation
    try:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        logger.error(f"Error checking PyTorch installation: {e}")
        return None, {}
    
    # Find model checkpoints
    checkpoint_path = None
    candidate_ckpts = [
        "outputs/final_fixed/best.pt",
        "outputs/expanded_dataset_fixed/best.pt",
        "outputs/uk_full_cpu/best.pt",
        "outputs/uk_top15_e50/best.pt"
    ]
    
    # Add any .pt file in the outputs directory
    import glob
    pt_files = glob.glob("outputs/**/*.pt", recursive=True)
    candidate_ckpts.extend([f for f in pt_files if f not in candidate_ckpts])
    
    # Find the first existing checkpoint
    for ckpt in candidate_ckpts:
        if os.path.exists(ckpt):
            checkpoint_path = os.path.abspath(ckpt)
            logger.info(f"✅ Found checkpoint: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        logger.error("❌ No model checkpoint found")
        return None, {}
    
    # Try to load the checkpoint
    try:
        logger.info(f"\n🔍 Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        logger.info("✅ Checkpoint loaded successfully")
        
        # Log checkpoint structure
        logger.info(f"Checkpoint keys: {list(ckpt.keys())}")
        if 'model' in ckpt:
            logger.info(f"Model state dict keys: {list(ckpt['model'].keys())[:5]}...")
        if 'cfg' in ckpt:
            logger.info(f"Config keys: {list(ckpt['cfg'].keys())}")
        
        return ckpt.get('model'), ckpt.get('cfg', {})
        
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {str(e)}", exc_info=True)
        return None, {}

if __name__ == "__main__":
    model, config = load_model()
    if model is not None:
        print("\n🎉 Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Config keys: {list(config.keys())}")
    else:
        print("\n❌ Failed to load model")
