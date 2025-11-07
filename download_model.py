#!/usr/bin/env python3
"""
Download the original DeepSeek V3 PyTorch weights from Hugging Face.
"""

from huggingface_hub import hf_hub_download
import os

def download_model():
    print("=" * 70)
    print("Downloading DeepSeek V3 PyTorch Weights")
    print("=" * 70)
    
    os.makedirs("model_weights", exist_ok=True)
    
    try:
        print("\nDownloading... (424 MB)")
        model_path = hf_hub_download(
            repo_id="Mayank022/DeepSeek-V3-from-Scratch",
            filename="best_deepseek_v3.pt",
            local_dir="model_weights",
            local_dir_use_symlinks=False
        )
        
        print("\n✅ Download complete!")
        print(f"Saved to: {model_path}")
        print("\nNext: Run python3 remap_weights_for_mlx_unified.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_model()
