"""
Debug script to analyze model architecture and checkpoint compatibility.
"""
import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_model.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the model class
try:
    sys.path.insert(0, str(Path(__file__).parent.absolute()))
    from src.model import CNNBiLSTMAttn, SimplifiedRiskModel
    logger.info("✅ Successfully imported model classes")
except Exception as e:
    logger.error(f"❌ Failed to import model classes: {e}")
    sys.exit(1)

def analyze_checkpoint(checkpoint_path):
    """Analyze the checkpoint file structure and content."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing checkpoint: {checkpoint_path}")
    logger.info(f"{'='*70}\n")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info("✅ Checkpoint loaded successfully")
        
        # Print checkpoint structure
        logger.info("\n=== Checkpoint Structure ===")
        for key, value in checkpoint.items():
            if key == 'model':
                logger.info(f"model: State dict with {len(value)} parameters")
            elif isinstance(value, dict):
                logger.info(f"{key}: Dict with keys: {list(value.keys())}")
            elif isinstance(value, (list, tuple)):
                logger.info(f"{key}: {type(value).__name__} of length {len(value)}")
            else:
                logger.info(f"{key}: {value}")
        
        # Get model state dict
        if 'model' not in checkpoint:
            logger.error("❌ No 'model' key in checkpoint")
            return None
            
        return checkpoint
        
    except Exception as e:
        logger.error(f"❌ Error analyzing checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def check_model_compatibility(checkpoint):
    """Check if the model architecture matches the checkpoint."""
    logger.info("\n=== Model Compatibility Check ===")
    
    # Get model config
    model_config = checkpoint.get('cfg', {}).get('model', {})
    num_features = len(checkpoint.get('features', []))
    
    logger.info(f"Number of features: {num_features}")
    logger.info(f"Model config: {model_config}")
    
    # Initialize model
    try:
        if 'final_fixed' in str(checkpoint_path) or 'expanded_dataset_fixed' in str(checkpoint_path):
            logger.info("Initializing SimplifiedRiskModel...")
            model = SimplifiedRiskModel(
                in_channels=num_features,
                hidden_dim=128,
                dropout=0.3
            )
        else:
            logger.info("Initializing CNNBiLSTMAttn...")
            model = CNNBiLSTMAttn(
                in_channels=num_features,
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
        
        logger.info("✅ Model initialized successfully")
        
        # Print model architecture
        logger.info("\n=== Model Architecture ===")
        for name, param in model.named_parameters():
            logger.info(f"{name}: {tuple(param.shape)}")
        
        # Try loading weights
        logger.info("\n=== Loading Weights ===")
        try:
            # First try strict loading
            model.load_state_dict(checkpoint['model'], strict=True)
            logger.info("✅ Weights loaded successfully with strict=True")
            return True
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.warning(f"⚠️ Size mismatch: {str(e)}")
                logger.info("Attempting to load with strict=False...")
                
                # Try with strict=False
                model.load_state_dict(checkpoint['model'], strict=False)
                logger.info("✅ Weights loaded with strict=False (some parameters may not have loaded)")
                
                # Check which parameters didn't load
                model_state_dict = model.state_dict()
                ckpt_state_dict = checkpoint['model']
                
                missing = [k for k in ckpt_state_dict.keys() if k not in model_state_dict]
                unexpected = [k for k in model_state_dict.keys() if k not in ckpt_state_dict]
                
                if missing:
                    logger.warning(f"Missing keys in model: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys in model: {unexpected}")
                
                return True if not missing else False
            else:
                logger.error(f"❌ Error loading weights: {str(e)}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error initializing model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Check for PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check checkpoint
    checkpoint_path = "outputs/uk_full_cpu/best.pt"
    if not os.path.exists(checkpoint_path):
        # Try to find any checkpoint
        for root, _, files in os.walk("outputs"):
            if "best.pt" in files:
                checkpoint_path = os.path.join(root, "best.pt")
                logger.info(f"Found checkpoint at: {checkpoint_path}")
                break
    
    if os.path.exists(checkpoint_path):
        checkpoint = analyze_checkpoint(checkpoint_path)
        if checkpoint:
            success = check_model_compatibility(checkpoint)
            if success:
                logger.info("\n✅ Model and checkpoint are compatible!")
            else:
                logger.error("\n❌ Model and checkpoint are not compatible")
    else:
        logger.error(f"No checkpoint found in {os.path.abspath('outputs')}")
        logger.info("\nAvailable checkpoints:")
        for root, dirs, files in os.walk("outputs"):
            for file in files:
                if file.endswith(".pt"):
                    logger.info(f"- {os.path.join(root, file)}")
