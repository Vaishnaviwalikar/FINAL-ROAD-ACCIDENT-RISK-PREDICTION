"""
Test script to diagnose model loading issues.
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_loading.log')
    ]
)
logger = logging.getLogger(__name__)

def test_model_loading(checkpoint_path):
    """Test loading a model checkpoint."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing model loading for: {checkpoint_path}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        logger.info("Loading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        logger.info("Checkpoint loaded successfully")
        
        # Print checkpoint structure
        logger.info(f"Checkpoint keys: {list(ckpt.keys())}")
        
        if 'model' not in ckpt:
            logger.error("No 'model' key in checkpoint")
            return False
            
        # Print some model keys
        model_keys = list(ckpt['model'].keys())
        logger.info(f"Model has {len(model_keys)} parameters")
        logger.info(f"First 5 params: {model_keys[:5]}")
        
        # Try to load the model
        logger.info("\nAttempting to load model...")
        from src.model import CNNBiLSTMAttn, SimplifiedRiskModel
        
        # Try to determine model type from checkpoint
        if 'final_fixed' in str(checkpoint_path) or 'expanded_dataset_fixed' in str(checkpoint_path):
            logger.info("Loading SimplifiedRiskModel")
            model = SimplifiedRiskModel(
                in_channels=len(ckpt.get('features', [])),
                hidden_dim=128,
                dropout=0.3
            )
        else:
            logger.info("Loading CNNBiLSTMAttn")
            model_config = ckpt.get('cfg', {}).get('model', {})
            model = CNNBiLSTMAttn(
                in_channels=len(ckpt.get('features', [])),
                cnn_channels=tuple(model_config.get("cnn_channels", [64, 128])),
                kernel_sizes=tuple(model_config.get("kernel_sizes", [3, 3])),
                pool_size=model_config.get("pool_size", 2),
                fc_dim=model_config.get("fc_dim", 128),
                attn_spatial_dim=model_config.get("attn_spatial_dim", 64),
                attn_temporal_dim=model_config.get("attn_temporal_dim", 64),
                lstm_hidden=model_config.get("lstm_hidden", 64),
                lstm_layers=model_config.get("lstm_layers", 1),
                dropout=model_config.get("dropout", 0.3),
            )
        
        # Try loading with strict=False first
        try:
            logger.info("Loading model weights with strict=False...")
            model.load_state_dict(ckpt['model'], strict=False)
            logger.info("✅ Model loaded with strict=False")
            
            # Check which keys didn't match
            model_state_dict = model.state_dict()
            missing = [k for k in ckpt['model'].keys() if k not in model_state_dict]
            unexpected = [k for k in model_state_dict.keys() if k not in ckpt['model']]
            
            if missing:
                logger.warning(f"Missing keys in model: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in model: {unexpected}")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in test_model_loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test with a specific checkpoint
    checkpoint_path = "outputs/uk_full_cpu/best.pt"
    if not os.path.exists(checkpoint_path):
        # Try to find any checkpoint
        for root, _, files in os.walk("outputs"):
            if "best.pt" in files:
                checkpoint_path = os.path.join(root, "best.pt")
                break
    
    if os.path.exists(checkpoint_path):
        success = test_model_loading(checkpoint_path)
        if success:
            logger.info("\n✅ Model loading test completed successfully!")
        else:
            logger.error("\n❌ Model loading test failed!")
    else:
        logger.error(f"No checkpoint found in {os.path.abspath('outputs')}")
        logger.info("\nAvailable checkpoints:")
        for root, dirs, files in os.walk("outputs"):
            for file in files:
                if file.endswith(".pt"):
                    logger.info(f"- {os.path.join(root, file)}")
