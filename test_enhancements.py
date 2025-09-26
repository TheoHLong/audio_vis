#!/usr/bin/env python3
"""Test the enhanced audio visualization pipeline with neuron sorting."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline import CometPipeline
from backend.config import PipelineConfig
from backend.neuron_analysis import NeuronAnalyzer

def test_neuron_analyzer():
    """Test the neuron analyzer with synthetic data."""
    print("Testing Neuron Analyzer...")
    
    analyzer = NeuronAnalyzer(n_components=3)
    
    # Create synthetic activations (time x neurons)
    n_time = 100
    n_neurons = 50
    
    # Create activations with different patterns
    activations = np.zeros((n_time, n_neurons))
    
    # Some neurons respond to pitch (sine wave pattern)
    for i in range(10):
        freq = 0.1 + i * 0.05
        activations[:, i] = np.sin(2 * np.pi * freq * np.arange(n_time))
    
    # Some neurons respond to energy (ramp pattern)
    for i in range(10, 20):
        activations[:, i] = np.linspace(0, 1, n_time) + np.random.normal(0, 0.1, n_time)
    
    # Some neurons have random activity
    for i in range(20, n_neurons):
        activations[:, i] = np.random.randn(n_time)
    
    # Create synthetic audio
    sample_rate = 16000
    duration = n_time * 0.01  # 10ms per frame
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Audio with varying pitch and amplitude
    pitch_freq = 200 + 50 * np.sin(2 * np.pi * 0.5 * t)
    audio = np.sin(2 * np.pi * pitch_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
    
    # Extract features
    features = analyzer.extract_audio_features(audio, sample_rate)
    
    print(f"  - Extracted {len(features)} audio features")
    for name, feat in features.items():
        print(f"    - {name}: shape {feat.shape}")
    
    # Analyze layer
    result = analyzer.analyze_layer('test_layer', activations, features)
    
    print(f"  - Sorted neurons: {len(result['sorted_indices'])} neurons")
    print(f"  - PC timeseries shape: {np.array(result['pc_timeseries']).shape}")
    print(f"  - Explained variance: {result['explained_variance']}")
    
    # Test stats computation
    stats = analyzer.compute_layer_stats(result['reordered_activations'])
    print(f"  - Computed stats: {list(stats.keys())}")
    
    print("✓ Neuron Analyzer test passed!\n")
    return True

def test_pipeline_integration():
    """Test the integrated pipeline with neuron analysis."""
    print("Testing Pipeline Integration...")
    
    config = PipelineConfig()
    pipeline = CometPipeline(config=config)
    
    # Generate test audio
    duration = 2.0
    sample_rate = config.sample_rate
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create speech-like audio with formants
    f1 = 700 + 200 * np.sin(2 * np.pi * 0.5 * t)  # First formant
    f2 = 1220 + 300 * np.sin(2 * np.pi * 0.3 * t)  # Second formant
    
    audio = (0.7 * np.sin(2 * np.pi * f1 * t) + 
             0.3 * np.sin(2 * np.pi * f2 * t))
    
    # Add amplitude modulation
    audio *= 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    
    # Process in chunks
    chunk_size = 1024
    payloads = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size].astype(np.float32)
        payload = pipeline.process_samples(chunk)
        if payload:
            payloads.append(payload)
    
    print(f"  - Processed {len(payloads)} payloads")
    
    if payloads:
        last_payload = payloads[-1]
        
        # Check for neuron analysis data
        for layer in last_payload.get('layers', []):
            layer_name = layer['name']
            if 'neuron_analysis' in layer:
                analysis = layer['neuron_analysis']
                print(f"  - Layer {layer_name} has neuron analysis:")
                print(f"    - Sorted indices: {len(analysis.get('sorted_indices', []))} neurons")
                print(f"    - PC timeseries available: {'pc_timeseries' in analysis}")
                print(f"    - Stats available: {'stats' in analysis}")
            else:
                print(f"  - Layer {layer_name}: No neuron analysis (insufficient data)")
    
    print("✓ Pipeline integration test passed!\n")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Enhanced Audio Visualization Pipeline")
    print("=" * 60 + "\n")
    
    try:
        # Test individual components
        test_neuron_analyzer()
        
        # Test integration
        test_pipeline_integration()
        
        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
