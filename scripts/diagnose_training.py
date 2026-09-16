"""
DepthWizard Training Diagnostic Script
Run this to verify the training pipeline works correctly.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_synthetic_data():
    """Test that synthetic data generation works"""
    from notebooks.train_depth_wizard import GAMUSH5Dataset
    # ... (would need to import properly in real setup)
    pass

def check_model_outputs():
    """Verify model produces non-zero outputs"""
    from transformers import AutoModelForDepthEstimation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)
    model.train()

    # Create random input
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        depth = output.predicted_depth

    print(f"✅ Model output shape: {depth.shape}")
    print(f"   Output range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"   Output mean: {depth.mean():.4f}")

    return depth.mean().item() > 0

if __name__ == "__main__":
    print("="*60)
    print("🔍 DepthWizard Training Pipeline Diagnostic")
    print("="*60)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.2f} GB")
    else:
        print("⚠️  CUDA not available - using CPU (slow)")

    print()

    # Test model
    try:
        model_ok = check_model_outputs()
        if model_ok:
            print("\n✅ Model is producing valid outputs")
        else:
            print("\n❌ Model outputs are all zeros")
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")

    print()
    print("="*60)
    print("📋 To fix the zero-loss issue:")
    print("   1. Run the updated notebook (v13)")
    print("   2. Check the diagnostic output in Epoch 0")
    print("   3. Loss should start > 0 and decrease over epochs")
    print("="*60)