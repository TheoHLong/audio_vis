#!/usr/bin/env python3
"""
Test script to verify Whisper model functionality
Run this script to test if Whisper is working correctly on your system.
"""

import numpy as np
import logging
import torch
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_whisper():
    """Test the Whisper model with a simple audio input"""
    
    print("=" * 60)
    print("Testing Whisper Model")
    print("=" * 60)
    
    # Check available devices
    print("\n🔧 System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Determine device
    if torch.cuda.is_available():
        device = 0
        print("\n✅ Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("\n✅ Using Apple Silicon (MPS)")
    else:
        device = -1
        print("\n✅ Using CPU")
    
    print("\n📥 Loading Whisper model...")
    try:
        # Initialize the pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=device,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=False,
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
                "max_new_tokens": 128,
            }
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("\n🎤 Testing with synthetic audio...")
    
    # Test 1: Silent audio
    print("\nTest 1: Silent audio")
    sample_rate = 16000
    duration = 1.0  # 1 second
    silent_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    try:
        result = pipe({"array": silent_audio, "sampling_rate": sample_rate})
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        text = result.get("text", "") if isinstance(result, dict) else ""
        print(f"Extracted text: '{text}'")
        print("✅ Silent audio test passed")
    except Exception as e:
        print(f"❌ Silent audio test failed: {e}")
    
    # Test 2: Random noise (simulates speech-like audio)
    print("\nTest 2: Random noise audio")
    noise_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    try:
        result = pipe({"array": noise_audio, "sampling_rate": sample_rate})
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        text = result.get("text", "") if isinstance(result, dict) else ""
        print(f"Extracted text: '{text}'")
        print("✅ Noise audio test passed")
    except Exception as e:
        print(f"❌ Noise audio test failed: {e}")
    
    # Test 3: Sine wave (tone)
    print("\nTest 3: Sine wave (440Hz tone)")
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    
    try:
        result = pipe({"array": tone_audio, "sampling_rate": sample_rate})
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        text = result.get("text", "") if isinstance(result, dict) else ""
        print(f"Extracted text: '{text}'")
        print("✅ Tone audio test passed")
    except Exception as e:
        print(f"❌ Tone audio test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    
    print("\n📌 Notes:")
    print("- Silent audio typically produces empty or minimal text")
    print("- Random noise might produce hallucinated text")
    print("- Sine waves usually produce empty text")
    print("- Real speech audio should produce actual transcription")
    
    return True

if __name__ == "__main__":
    success = test_whisper()
    exit(0 if success else 1)