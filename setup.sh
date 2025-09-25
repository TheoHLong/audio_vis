#!/bin/bash
# Setup script for audio visualization app

echo "=========================================="
echo "AUDIO VISUALIZATION SETUP"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Install/upgrade pip
echo -e "\n📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install required packages
echo -e "\n📦 Installing required packages..."
python3 -m pip install -r requirements.txt

# Install Whisper-specific dependencies
echo -e "\n🎤 Installing Whisper dependencies..."
python3 -m pip install transformers>=4.36.0 torch torchaudio

# Try to initialize Whisper model
echo -e "\n🎤 Pre-downloading Whisper model..."
python3 init_whisper.py

if [ $? -eq 0 ]; then
    echo -e "\n✅ Setup complete! Whisper model is ready."
    echo "You can now start the app with: python3 start_app.py"
else
    echo -e "\n⚠️ Whisper model initialization failed, but you can still try running the app."
    echo "Start the app with: python3 start_app.py"
fi