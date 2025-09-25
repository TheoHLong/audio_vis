#!/usr/bin/env python3
"""
Pre-download and initialize Whisper model before starting the app
This ensures the model is ready when the app starts
"""

import logging
import sys
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_whisper():
    """Download and initialize Whisper model"""
    logger.info("=" * 60)
    logger.info("PRE-INITIALIZING WHISPER MODEL")
    logger.info("=" * 60)
    
    try:
        # Check PyTorch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Download and initialize model
        logger.info("Downloading openai/whisper-tiny.en if needed...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=-1  # Use CPU for compatibility
        )
        
        # Test it
        import numpy as np
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = pipe({"array": test_audio, "sampling_rate": 16000})
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"Test result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Whisper: {e}")
        return False

if __name__ == "__main__":
    success = initialize_whisper()
    if success:
        logger.info("\n✅ Whisper model is ready!")
        logger.info("You can now start the main application.")
    else:
        logger.error("\n❌ Failed to initialize Whisper model!")
        logger.error("Please check the error messages above.")
        sys.exit(1)