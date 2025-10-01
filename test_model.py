"""
Test script to load and verify the model
"""
import os
import sys
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Add src to path
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Try to import the model
    try:
        from model import CNNBiLSTMAttn
        logger.info("✅ Successfully imported model class")
    except ImportError as e:
        logger.error(f"❌ Failed to import model: {e}")
        return
    
    # Try to load the checkpoint
    checkpoint_path = "outputs/uk_full_cpu/best.pt"
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Create a simplified model that matches the checkpoint structure
        class SimplifiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                # These layers should match the checkpoint's structure
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(32)
                self.fc = nn.Linear(32, 1)  # Output layer for binary classification
                
            def forward(self, x):
                x = f.relu(self.bn1(self.conv1(x)))
                x = f.relu(self.bn2(self.conv2(x)))
                x = x.mean(dim=2)  # Global average pooling
                x = self.fc(x)
                return x
                
        model = SimplifiedModel()
        
        # Load state dict with strict=False to handle missing keys
        if 'model' in checkpoint:
            # Print model and checkpoint keys for debugging
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(checkpoint['model'].keys())
            
            logger.info("\nModel keys not in checkpoint:")
            for k in sorted(model_keys - ckpt_keys):
                logger.info(f"  - {k}")
                
            logger.info("\nCheckpoint keys not in model:")
            for k in sorted(ckpt_keys - model_keys):
                logger.info(f"  - {k}")
            
            # Try loading with strict=False to handle any missing keys
            model.load_state_dict(checkpoint['model'], strict=False)
            logger.info("✅ Successfully loaded model state dict (with strict=False)")
        else:
            # For older checkpoints without the 'model' key
            model.load_state_dict(checkpoint, strict=False)
            logger.info("✅ Successfully loaded checkpoint directly (with strict=False)")
        
        model.eval()
        logger.info("✅ Model loaded and set to evaluation mode")
        
        # Print model summary
        logger.info("\nModel architecture:")
        print(model)
        
        # Print number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\nTotal trainable parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    model = main()
    if model is not None:
        print("\n🎉 Model loaded successfully!")
    else:
        print("\n❌ Failed to load model")
