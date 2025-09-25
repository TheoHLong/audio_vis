#!/usr/bin/env python3
"""
Debug script to check Whisper model status and fix common issues
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_environment():
    """Check the environment and dependencies"""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if we can import required libraries
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"   MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Run: pip install torch")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers installed: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed!")
        print("   Run: pip install transformers")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy installed: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed!")
        return False
    
    # Check cache directory
    cache_dir = Path.home() / ".cache" / "huggingface"
    print(f"\nHuggingFace cache directory: {cache_dir}")
    print(f"Cache directory exists: {cache_dir.exists()}")
    
    # Check if model is already cached
    model_cache = cache_dir / "hub"
    if model_cache.exists():
        models = list(model_cache.glob("models--openai--whisper*"))
        if models:
            print(f"✅ Found cached Whisper models:")
            for model in models:
                print(f"   - {model.name}")
        else:
            print("⚠️ No Whisper models found in cache")
    
    return True

def test_whisper_simple():
    """Test Whisper with the simplest possible configuration"""
    print("\n" + "=" * 60)
    print("TESTING WHISPER MODEL")
    print("=" * 60)
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        import numpy as np
        
        print("Loading Whisper model components...")
        
        # Load processor and model separately for better control
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        print("✅ Processor loaded")
        
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        print("✅ Model loaded")
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.01
        
        print(f"\nProcessing test audio ({duration}s @ {sample_rate}Hz)...")
        
        # Process audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features, max_new_tokens=50)
        
        # Decode
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"✅ Test transcription successful!")
        print(f"   Result: '{transcription}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_pipeline():
    """Test Whisper using the pipeline API"""
    print("\n" + "=" * 60)
    print("TESTING WHISPER PIPELINE")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        import numpy as np
        
        print("Creating ASR pipeline...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=-1  # Force CPU
        )
        print("✅ Pipeline created")
        
        # Test with silence
        print("\nTest 1: Silent audio")
        audio = np.zeros(16000, dtype=np.float32)
        result = pipe({"array": audio, "sampling_rate": 16000})
        print(f"   Result: {result}")
        
        # Test with noise
        print("\nTest 2: Random noise")
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        result = pipe({"array": audio, "sampling_rate": 16000})
        print(f"   Result: {result}")
        
        print("\n✅ Pipeline tests successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks"""
    print("WHISPER MODEL DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Test simple model loading
    if test_whisper_simple():
        print("\n✅ Direct model loading works!")
    else:
        print("\n⚠️ Direct model loading failed, but pipeline might still work.")
    
    # Test pipeline
    if test_whisper_pipeline():
        print("\n✅ Pipeline API works!")
        print("\n" + "=" * 60)
        print("DIAGNOSIS: Whisper should work in your app!")
        print("If it still shows 'warming', check the app logs for errors.")
        print("=" * 60)
    else:
        print("\n❌ Pipeline API failed.")
        print("\nPossible solutions:")
        print("1. Reinstall transformers: pip install --upgrade transformers")
        print("2. Clear cache: rm -rf ~/.cache/huggingface")
        print("3. Check internet connection for model download")

if __name__ == "__main__":
    main()