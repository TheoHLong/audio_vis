#!/usr/bin/env python3
"""
Start the audio visualization app with proper initialization
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    logger.info("=" * 60)
    logger.info("AUDIO VISUALIZATION STARTUP")
    logger.info("=" * 60)
    
    # Step 1: Initialize Whisper model
    logger.info("\n📥 Step 1: Initializing Whisper model...")
    result = subprocess.run([sys.executable, "init_whisper.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Failed to initialize Whisper model!")
        logger.error(result.stderr)
        logger.info("\nTrying to start anyway...")
    else:
        logger.info("✅ Whisper model initialized successfully!")
    
    # Step 2: Start the FastAPI application
    logger.info("\n🚀 Step 2: Starting FastAPI application...")
    logger.info("Server will be available at: http://localhost:8000")
    logger.info("Press Ctrl+C to stop the server\n")
    
    # Start uvicorn with detailed logging
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ])

if __name__ == "__main__":
    main()