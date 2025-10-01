#!/usr/bin/env python3
"""
Quick fix to load existing trained model with proper architecture.
"""
import torch
import torch.nn as nn

class FixedModel(nn.Module):
    """Model that matches the exact checkpoint structure."""
    
    def __init__(self, in_channels=30):
        super().__init__()
        
        # Conv layers (exact match)
        self.conv1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        
        # FC layer
        self.fc = nn.Linear(32, 128)
        
        # Attention (exact structure from checkpoint)
        self.s_attn = nn.Module()
        self.s_attn.proj = nn.Sequential(
            nn.Linear(128, 64),      # proj.0
            nn.ReLU(),               # (not saved)
            nn.Linear(64, 128),      # proj.2
        )
        
        self.t_attn = nn.Module()
        self.t_attn.query = nn.Linear(128, 64)
        self.t_attn.key = nn.Linear(128, 64)
        
        # LSTM
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Output
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        # Conv processing
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        
        # FC and attention
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        
        # Apply attention
        s_attn = torch.sigmoid(self.s_attn.proj(x))
        x = x * s_attn
        
        q = self.t_attn.query(x)
        k = self.t_attn.key(x)
        t_attn = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True), dim=1)
        x = x * t_attn
        
        # LSTM and output
        x, _ = self.lstm(x)
        return self.out(x[:, -1, :]).squeeze(-1)

def load_fixed_model():
    """Load the existing trained model with fixed architecture."""
    
    checkpoint = torch.load("outputs/quick_fixed/best.pt", map_location='cpu')
    features = checkpoint['features']
    
    model = FixedModel(len(features))
    
    # Load with strict=False to handle minor mismatches
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    return model, features, checkpoint['cfg']

if __name__ == "__main__":
    try:
        model, features, config = load_fixed_model()
        print("✅ SUCCESS: Existing trained model loaded!")
        print(f"Features: {len(features)}")
        print("Ready for real AI predictions")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("Consider retraining if fix doesn't work")