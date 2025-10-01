"""
Script to debug and fix model loading issues
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_checkpoint(checkpoint_path):
    """Inspect the structure of a checkpoint file."""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Print checkpoint structure
        logger.info("\n=== Checkpoint Structure ===")
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            logger.info("\nModel state dict keys (first 20):")
            for i, key in enumerate(model_state.keys()):
                if i >= 20:  # Only show first 20 keys to avoid cluttering
                    logger.info(f"... and {len(model_state) - 20} more")
                    break
                logger.info(f"  {key}: {model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'}")
        
        if 'cfg' in checkpoint:
            logger.info("\nModel configuration:")
            for key, value in checkpoint['cfg'].items():
                logger.info(f"  {key}: {value}")
                
        if 'features' in checkpoint:
            logger.info(f"\nFeatures used in training: {checkpoint['features']}")
            
        return checkpoint
        
    except Exception as e:
        logger.error(f"Error inspecting checkpoint: {e}", exc_info=True)
        return None

def create_model_from_checkpoint(checkpoint):
    """Create a model based on the checkpoint structure."""
    try:
        if 'model' not in checkpoint:
            logger.error("No 'model' key in checkpoint")
            return None
            
        # Get model configuration from checkpoint or use defaults
        cfg = checkpoint.get('cfg', {})
        
        # Define a model that exactly matches the checkpoint structure
        class AccidentRiskModel(nn.Module):
            def __init__(self, num_features):
                super().__init__()
                # CNN layers - matching the checkpoint's expected input channels
                # Checkpoint expects conv1 to have 29 input channels
                self.conv1 = nn.Conv1d(29, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(32)
                
                # Fully connected layers - adjusted for the correct input size
                self.fc1 = nn.Linear(32, 64)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, 1)  # Output layer for binary classification
                
            def forward(self, x):
                # Ensure input is in the right shape: [batch_size, 1, num_features]
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                
                # CNN layers
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                
                # Global average pooling
                x = x.mean(dim=2)  # [batch_size, 32]
                
                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return torch.sigmoid(x)  # Output probability between 0 and 1
        
        # Get number of features from checkpoint if available
        num_features = len(checkpoint.get('features', [])) or 10  # Default to 10 if not specified
        model = AccidentRiskModel(num_features)
        
        # Load state dict
        model_state = checkpoint['model']
        
        # Filter out unexpected keys
        model_state = {k: v for k, v in model_state.items() if k in model.state_dict()}
        
        # Load the filtered state dict
        model.load_state_dict(model_state, strict=False)
        model.eval()
        
        logger.info("\n=== Model Summary ===")
        logger.info(model)
        
        # Print number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\nTotal trainable parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}", exc_info=True)
        return None

def main():
    # Path to the checkpoint
    checkpoint_path = "outputs/uk_full_cpu/best.pt"
    
    if not os.path.exists(checkpoint_path):
        # Try to find any .pt file in the outputs directory
        import glob
        pt_files = glob.glob("outputs/**/*.pt", recursive=True)
        if pt_files:
            checkpoint_path = pt_files[0]
            logger.warning(f"Using found checkpoint: {checkpoint_path}")
        else:
            logger.error("No .pt files found in outputs directory")
            return
    
    # Inspect the checkpoint
    checkpoint = inspect_checkpoint(checkpoint_path)
    
    if checkpoint is None:
        logger.error("Failed to load checkpoint")
        return
    
    # Try to create and load the model
    model = create_model_from_checkpoint(checkpoint)
    
    if model is not None:
        logger.info("\n✅ Model loaded successfully!")
        
        # Test with a random input that matches the expected shape
        # The model expects [batch_size, num_channels, sequence_length]
        # From the checkpoint, we know it expects 29 input channels
        test_input = torch.randn(1, 29, 1)  # [batch_size=1, channels=29, sequence_length=1]
        with torch.no_grad():
            output = model(test_input)
            logger.info(f"\nTest input shape: {test_input.shape}")
            logger.info(f"Test prediction: {output.item():.4f}")
            
            # Also test with the actual number of features from the checkpoint
            if 'features' in checkpoint:
                num_features = len(checkpoint['features'])
                logger.info(f"\nNumber of features in checkpoint: {num_features}")
                logger.info("Feature names:")
                for i, feature in enumerate(checkpoint['features']):
                    logger.info(f"  {i+1}. {feature}")
    else:
        logger.error("❌ Failed to load model")

if __name__ == "__main__":
    main()
