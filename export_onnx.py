"""
Exports PyTorch VoiceAntiSpoofCNN model to standard ONNX representation.
Input:  'input_mel_spectrogram' of shape (1, 1, 64, 300)
Output: 'logits' of shape (1, 2)
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_ROOT = os.path.dirname(CURRENT_DIR)
if AI_MODEL_ROOT not in sys.path:
    sys.path.insert(0, AI_MODEL_ROOT)

import torch
from training.model import VoiceAntiSpoofCNN

def export_to_onnx():
    checkpoint_path = os.path.join(AI_MODEL_ROOT, "saved_models", "best_baseline_cnn.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    onnx_output_path = os.path.join(AI_MODEL_ROOT, "saved_models", "voice_anti_spoof.onnx")
    
    # 1. Load Model
    model = VoiceAntiSpoofCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 2. Dummy Input (Batch=1, Channels=1, Mels=64, Frames=300)
    dummy_input = torch.randn(1, 1, 64, 300, dtype=torch.float32)
    
    # 3. Export using stable opset with embedded weights
    print(f"Exporting PyTorch model to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "input_mel_spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )
    
    file_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    print(f"ONNX Export Successful! File Size: {file_size_mb:.2f} MB ({onnx_output_path})")
    return onnx_output_path

if __name__ == "__main__":
    export_to_onnx()
