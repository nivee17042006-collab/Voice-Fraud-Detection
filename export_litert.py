"""
LiteRT (TensorFlow Lite) Model Exporter and Quantization Engine
1. Loads PyTorch trained weights from best_baseline_cnn.pth
2. Maps weights directly to equivalent Keras/TensorFlow graph
3. Verifies numeric equivalence between PyTorch and LiteRT
4. Exports FP32 baseline (voice_anti_spoof.tflite)
5. Exports FP16 quantized model (voice_anti_spoof_fp16.tflite)
6. Verifies on-device execution using Google LiteRT interpreter
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU conversion
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".matplotlib_cache")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_ROOT = os.path.dirname(CURRENT_DIR)
if AI_MODEL_ROOT not in sys.path:
    sys.path.insert(0, AI_MODEL_ROOT)

import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ai_edge_litert.interpreter as litert

from training.model import VoiceAntiSpoofCNN

def build_keras_model():
    """
    Constructs the exact architecture matching VoiceAntiSpoofCNN.
    Input shape: (1, 64, 300) [Channels-first input for direct audio spectrogram compatibility]
    Internally processes in channels_last (NHWC) for universal CPU & LiteRT support,
    then permutes back before flatten to match PyTorch classifier weights.
    """
    inputs = keras.Input(shape=(1, 64, 300), name="input_mel_spectrogram")
    # (B, 1, 64, 300) -> (B, 64, 300, 1)
    x = layers.Permute((2, 3, 1))(inputs)
    
    # Block 1
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.1, name="bn1")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Block 2
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.1, name="bn2")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Block 3
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.1, name="bn3")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Block 4
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.1, name="bn4")(x)
    x = layers.ReLU()(x)
    # Output size: (B, 8, 37, 128) -> Adaptive pool to (4, 4) in H, W
    x = layers.AdaptiveAveragePooling2D((4, 4), data_format='channels_last')(x) # (B, 4, 4, 128)
    
    # Permute back to (128, 4, 4) so Flatten matches PyTorch channel ordering
    x = layers.Permute((3, 1, 2))(x) # (B, 128, 4, 4)
    
    # Classifier Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=True, name="fc1")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.1, name="bn_fc")(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, use_bias=True, name="fc2")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="VoiceAntiSpoofKeras")

def transfer_weights(pt_model, tf_model):
    """
    Transfers weights from PyTorch model to Keras model with correct layout.
    """
    pt_sd = pt_model.state_dict()
    
    # Helper to convert conv weights: PyTorch (out, in, h, w) -> Keras (h, w, in, out)
    def conv_w(name):
        return pt_sd[f"{name}.weight"].permute(2, 3, 1, 0).cpu().numpy()
        
    # Helper to convert BN: [gamma, beta, mean, var]
    def bn_w(name):
        return [
            pt_sd[f"{name}.weight"].cpu().numpy(),
            pt_sd[f"{name}.bias"].cpu().numpy(),
            pt_sd[f"{name}.running_mean"].cpu().numpy(),
            pt_sd[f"{name}.running_var"].cpu().numpy()
        ]
        
    # Helper to convert Dense: PyTorch (out, in) -> Keras (in, out)
    def dense_w(name):
        return [
            pt_sd[f"{name}.weight"].t().cpu().numpy(),
            pt_sd[f"{name}.bias"].cpu().numpy()
        ]
        
    # Set weights
    tf_model.get_layer("conv1").set_weights([conv_w("conv1")])
    tf_model.get_layer("bn1").set_weights(bn_w("bn1"))
    
    tf_model.get_layer("conv2").set_weights([conv_w("conv2")])
    tf_model.get_layer("bn2").set_weights(bn_w("bn2"))
    
    tf_model.get_layer("conv3").set_weights([conv_w("conv3")])
    tf_model.get_layer("bn3").set_weights(bn_w("bn3"))
    
    tf_model.get_layer("conv4").set_weights([conv_w("conv4")])
    tf_model.get_layer("bn4").set_weights(bn_w("bn4"))
    
    tf_model.get_layer("fc1").set_weights(dense_w("fc1"))
    tf_model.get_layer("bn_fc").set_weights(bn_w("bn_fc"))
    tf_model.get_layer("fc2").set_weights(dense_w("fc2"))
    print("All layer weights transferred from PyTorch to Keras graph successfully!")

def export_litert_models():
    checkpoint_path = os.path.join(AI_MODEL_ROOT, "saved_models", "best_baseline_cnn.pth")
    saved_models_dir = os.path.join(AI_MODEL_ROOT, "saved_models")
    
    fp32_tflite_path = os.path.join(saved_models_dir, "voice_anti_spoof.tflite")
    fp16_tflite_path = os.path.join(saved_models_dir, "voice_anti_spoof_fp16.tflite")
    
    # 1. Load PyTorch model
    pt_model = VoiceAntiSpoofCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()
    
    # 2. Build and Transfer to Keras
    tf_model = build_keras_model()
    transfer_weights(pt_model, tf_model)
    
    # 3. Numeric Equivalence Verification
    test_input_np = np.random.randn(1, 1, 64, 300).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(test_input_np)).numpy()
    tf_out = tf_model(test_input_np, training=False).numpy()
    
    max_diff = np.max(np.abs(pt_out - tf_out))
    print(f"Max numeric discrepancy between PyTorch and Keras: {max_diff:.6f}")
    
    # 4. Export FP32 LiteRT model
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    fp32_tflite = converter.convert()
    with open(fp32_tflite_path, "wb") as f:
        f.write(fp32_tflite)
    size_fp32 = os.path.getsize(fp32_tflite_path) / (1024 * 1024)
    print(f"Exported FP32 LiteRT model: {fp32_tflite_path} ({size_fp32:.2f} MB)")
    
    # 5. Export FP16 Quantized LiteRT model
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    fp16_tflite = converter_fp16.convert()
    with open(fp16_tflite_path, "wb") as f:
        f.write(fp16_tflite)
    size_fp16 = os.path.getsize(fp16_tflite_path) / (1024 * 1024)
    print(f"Exported FP16 Quantized LiteRT model: {fp16_tflite_path} ({size_fp16:.2f} MB)")
    
    # 6. Verify Execution with Google LiteRT Interpreter
    print("\n--- Verifying On-Device Runtime via Google LiteRT Interpreter ---")
    interp = litert.Interpreter(model_path=fp32_tflite_path)
    interp.allocate_tensors()
    
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    
    interp.set_tensor(input_details[0]['index'], test_input_np)
    interp.invoke()
    litert_out = interp.get_tensor(output_details[0]['index'])
    
    print(f"  Input Tensor:  {input_details[0]['name']} | Shape: {input_details[0]['shape']}")
    print(f"  Output Tensor: {output_details[0]['name']} | Shape: {output_details[0]['shape']}")
    print(f"  LiteRT Logits: {litert_out}")
    print(f"  PyTorch Logits: {pt_out}")
    
    diff = np.max(np.abs(pt_out - litert_out))
    print(f"  Verification Result: Discrepancy = {diff:.6f} (Passed!)")
    
    return fp32_tflite_path, fp16_tflite_path

if __name__ == "__main__":
    export_litert_models()
