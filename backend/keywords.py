from __future__ import annotations

import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class KeywordCandidate:
    text: str
    confidence: float
    timestamp: float
    expires_at: float


@dataclass
class KeywordExtractor:
    sample_rate: int
    window_seconds: float = 0.9
    stride_seconds: float = 0.45
    max_keywords: int = 3

    _buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), init=False)
    _tokens: List[KeywordCandidate] = field(default_factory=list, init=False)
    _transcript_history: List[Tuple[float, str]] = field(default_factory=list, init=False)
    last_transcript: str = field(default="", init=False)
    _prev_transcript: str = field(default="", init=False)
    _last_emit: float = field(default=0.0, init=False)
    _pipeline: Optional[object] = field(default=None, init=False)
    _ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        disable = math.isclose(self.window_seconds, 0.0)
        if disable:
            logger.info("KeywordExtractor disabled by configuration (window_seconds=0)")
            return
        
        logger.info("Initializing Whisper model for keyword extraction...")
        
        try:
            # Import required modules
            import torch
            from transformers import pipeline
            
            logger.info(f"PyTorch version: {torch.__version__}")
            
            # Detect best available device - simplified
            device = -1  # Default to CPU
            device_name = "CPU"
            
            try:
                if torch.cuda.is_available():
                    device = 0
                    device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                    device_name = "Apple Silicon (MPS)"
            except Exception as e:
                logger.warning(f"Device detection issue: {e}, using CPU")
                device = -1
                device_name = "CPU"
            
            logger.info(f"Selected device: {device_name}")
            
            # Initialize the pipeline with simpler configuration
            logger.info("Loading openai/whisper-tiny.en model...")
            
            try:
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en",
                    device=device,
                    torch_dtype=torch.float32
                )
                
                logger.info("Whisper model loaded successfully!")
                
                # Quick test without audio to verify it's loaded
                logger.info("Testing Whisper model initialization...")
                # Create a very short test audio (0.1 seconds of silence)
                test_samples = int(self.sample_rate * 0.1)
                test_audio = np.zeros(test_samples, dtype=np.float32)
                
                # Try a simple transcription
                test_result = self._pipeline(
                    {"array": test_audio, "sampling_rate": self.sample_rate},
                    generate_kwargs={"max_new_tokens": 10}
                )
                
                logger.info(f"Whisper test completed. Result type: {type(test_result)}")
                if isinstance(test_result, dict):
                    logger.info(f"Test transcription: '{test_result.get('text', '')}'")
                
                self._ready = True
                logger.info("✅ KeywordExtractor is ready and operational!")
                
            except Exception as model_error:
                logger.error(f"Failed to load Whisper model: {model_error}")
                logger.info("Attempting fallback initialization...")
                
                # Try a simpler initialization
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en",
                    device=-1  # Force CPU
                )
                
                self._ready = True
                logger.info("✅ KeywordExtractor initialized with CPU fallback")
                
        except ImportError as import_error:
            logger.error(f"Missing required dependencies: {import_error}")
            logger.error("Please install transformers and torch: pip install transformers torch")
            self._pipeline = None
            self._ready = False
            
        except Exception as exc:
            logger.error(f"Unexpected error during Whisper initialization: {exc}", exc_info=True)
            self._pipeline = None
            self._ready = False
        
        # Log final status
        if self._ready:
            logger.info("=" * 60)
            logger.info("WHISPER STATUS: ✅ ONLINE")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("WHISPER STATUS: ❌ FAILED TO INITIALIZE")
            logger.error("Check the logs above for details")
            logger.error("=" * 60)

    @property
    def window_size(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def stride_size(self) -> int:
        return int(self.sample_rate * self.stride_seconds)

    def append(self, samples: np.ndarray) -> None:
        if not self._ready:
            return
        self._buffer = np.concatenate([self._buffer, samples])
        # Cap buffer to 2 * window to avoid runaway memory
        max_len = self.window_size * 2
        if self._buffer.shape[0] > max_len:
            self._buffer = self._buffer[-max_len:]

    def maybe_extract(self, timestamp: float) -> List[KeywordCandidate]:
        if not self._ready:
            return []
        
        if self._buffer.shape[0] < self.window_size:
            return []
        
        if timestamp - self._last_emit < self.stride_seconds:
            return []
        
        window = self._buffer[-self.window_size:]
        
        # Check if audio has any content
        max_val = np.abs(window).max()
        if max_val < 1e-6:  # Nearly silent
            self._last_emit = timestamp
            return self._tokens[-self.max_keywords:] if self._tokens else []
        
        # Normalize audio if needed
        if max_val < 0.01:  # Very quiet audio
            window = window / max_val * 0.1
        
        try:
            # Simple transcription call
            result = self._pipeline(
                {"array": window, "sampling_rate": self.sample_rate},
                generate_kwargs={
                    "max_new_tokens": 128,
                    "language": "en",
                    "task": "transcribe"
                }
            )
            
            # Extract text from result
            if isinstance(result, dict):
                text = result.get("text", "")
            elif isinstance(result, list) and len(result) > 0:
                text = result[0].get("text", "")
            else:
                text = str(result) if result else ""
            
            # Log successful transcriptions
            if text and text.strip():
                logger.info(f"🎤 Transcribed: '{text.strip()}'")
            
        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            text = ""
        
        cleaned = text.strip()
        if cleaned and cleaned != self._prev_transcript:
            self._transcript_history.append((timestamp, cleaned))
            self._prev_transcript = cleaned
        
        # Clean up old transcript history
        cutoff = timestamp - 12.0
        if cutoff > 0:
            self._transcript_history = [item for item in self._transcript_history if item[0] >= cutoff]
        
        # Update the full transcript
        self.last_transcript = " ".join(segment for _, segment in self._transcript_history).strip()
        
        # Extract keywords from the text
        keywords = self._extract_keywords(text)
        self._last_emit = timestamp
        expires_at = timestamp + 2.0
        
        candidates = [
            KeywordCandidate(text=word, confidence=conf, timestamp=timestamp, expires_at=expires_at)
            for word, conf in keywords
        ]
        
        self._tokens.extend(candidates)
        # Keep only recent tokens
        self._tokens = [token for token in self._tokens if token.expires_at > timestamp]
        # Return the latest ones (max_keywords)
        return self._tokens[-self.max_keywords:]

    def _extract_keywords(self, text: str) -> List[tuple[str, float]]:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        unique = []
        seen = set()
        for word in words[::-1]:  # reverse to keep latest words first
            if word in seen:
                continue
            seen.add(word)
            if len(word) < 3:
                continue
            score = min(1.0, 0.5 + len(word) / 8.0)
            unique.append((word, score))
            if len(unique) >= self.max_keywords:
                break
        return unique