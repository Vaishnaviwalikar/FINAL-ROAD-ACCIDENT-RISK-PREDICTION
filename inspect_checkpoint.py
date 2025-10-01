"""
Script to inspect the model checkpoint structure.
"""
import os
import torch
import json

def main():
    checkpoint_path = "outputs/uk_full_cpu/best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\n=== Checkpoint Keys ===")
        print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
        
        # Check if 'model' key exists
        if 'model' not in checkpoint:
            print("\nError: 'model' key not found in checkpoint")
            return
        
        # Print model state dict info
        model_state_dict = checkpoint['model']
        print(f"\n=== Model State Dict ===")
        print(f"Number of parameters: {len(model_state_dict)}")
        
        # Print first 10 parameter shapes
        print("\nFirst 10 parameters:")
        for i, (name, param) in enumerate(model_state_dict.items()):
            if i >= 10:
                print(f"... and {len(model_state_dict) - 10} more parameters")
                break
            print(f"  {name}: {tuple(param.shape)}")
        
        # Print config if available
        if 'cfg' in checkpoint:
            print("\n=== Model Config ===")
            print(json.dumps(checkpoint['cfg'], indent=2))
        
        # Print features if available
        if 'features' in checkpoint:
            print("\n=== Features ===")
            print(f"Number of features: {len(checkpoint['features'])}")
            print(f"First 10 features: {checkpoint['features'][:10]}")
        
    except Exception as e:
        print(f"\nError loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
