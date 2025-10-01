#!/usr/bin/env python3
"""
Ensure proper UTF-8 encoding for .env file
"""
import os

def fix_env_encoding():
    env_path = '.env'
    if not os.path.exists(env_path):
        print("❌ .env file not found")
        return False
    
    try:
        # Read with binary mode to avoid encoding issues
        with open(env_path, 'rb') as f:
            content = f.read()
        
        # Write back with explicit UTF-8 encoding
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content.decode('utf-8'))
            
        print("✅ .env encoding fixed to UTF-8")
        return True
    except Exception as e:
        print(f"❌ Error fixing encoding: {e}")
        return False

if __name__ == "__main__":
    fix_env_encoding()
