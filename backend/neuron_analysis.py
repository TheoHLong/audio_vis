"""
Neuron analysis module for sorting neurons by their tuning properties 
and extracting principal components for visualization.
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeuronTuning:
    """Stores tuning properties for a single neuron."""
    neuron_idx: int
    pitch_correlation: float
    energy_correlation: float
    spectral_correlation: float
    primary_feature: str
    cluster_id: int = -1


class NeuronAnalyzer:
    """Analyzes and sorts neurons by their tuning properties."""
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca_models = {}
        self.neuron_order = {}
        self.neuron_tunings = {}
        self.pc_timeseries = {}
        
    def extract_audio_features(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, np.ndarray]:
        """Extract acoustic features from audio for correlation analysis."""
        features = {}
        
        # Energy/RMS
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Compute energy
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy.append(np.sqrt(np.mean(frame**2)))
        features['energy'] = np.array(energy)
        
        # Estimate pitch using autocorrelation
        pitch_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            pitch = self._estimate_pitch_autocorr(frame, sample_rate)
            pitch_values.append(pitch)
        features['pitch'] = np.array(pitch_values)
        
        # Spectral centroid
        spectral_centroids = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            centroid = self._compute_spectral_centroid(frame, sample_rate)
            spectral_centroids.append(centroid)
        features['spectral_centroid'] = np.array(spectral_centroids)
        
        return features
    
    def _estimate_pitch_autocorr(self, frame: np.ndarray, sample_rate: int) -> float:
        """Simple pitch estimation using autocorrelation."""
        # Remove DC component
        frame = frame - np.mean(frame)
        
        # Compute autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after the zero-lag peak
        min_period = int(sample_rate / 500)  # 500 Hz max
        max_period = int(sample_rate / 50)   # 50 Hz min
        
        if max_period < len(corr):
            corr_segment = corr[min_period:max_period]
            if len(corr_segment) > 0 and np.max(corr_segment) > 0:
                peak_idx = np.argmax(corr_segment) + min_period
                pitch = sample_rate / peak_idx
                return pitch
        return 0.0
    
    def _compute_spectral_centroid(self, frame: np.ndarray, sample_rate: int) -> float:
        """Compute spectral centroid of a frame."""
        # Apply window
        window = np.hanning(len(frame))
        frame = frame * window
        
        # Compute FFT
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)
        
        # Compute frequencies
        freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)
        
        # Compute centroid
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return centroid
        return 0.0
    
    def analyze_layer(self, 
                     layer_name: str,
                     activations: np.ndarray, 
                     audio_features: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze a layer's neurons and sort by tuning properties.
        
        Args:
            layer_name: Name of the layer (e.g., 'L2', 'L6', 'L10')
            activations: Shape (time_steps, n_neurons)
            audio_features: Dictionary of audio features
            
        Returns:
            Dictionary with sorted neuron indices and analysis results
        """
        n_time, n_neurons = activations.shape
        
        # Ensure feature alignment
        min_len = min(n_time, len(audio_features['energy']))
        activations = activations[:min_len]
        
        tunings = []
        
        for neuron_idx in range(n_neurons):
            neuron_activity = activations[:, neuron_idx]
            
            # Compute correlations with audio features
            pitch_corr = 0.0
            if np.std(audio_features['pitch'][:min_len]) > 0:
                pitch_corr, _ = pearsonr(neuron_activity, audio_features['pitch'][:min_len])
                
            energy_corr = 0.0
            if np.std(audio_features['energy'][:min_len]) > 0:
                energy_corr, _ = pearsonr(neuron_activity, audio_features['energy'][:min_len])
                
            spectral_corr = 0.0
            if np.std(audio_features['spectral_centroid'][:min_len]) > 0:
                spectral_corr, _ = pearsonr(neuron_activity, audio_features['spectral_centroid'][:min_len])
            
            # Determine primary feature
            correlations = {
                'pitch': abs(pitch_corr) if not np.isnan(pitch_corr) else 0,
                'energy': abs(energy_corr) if not np.isnan(energy_corr) else 0,
                'spectral': abs(spectral_corr) if not np.isnan(spectral_corr) else 0
            }
            primary_feature = max(correlations, key=correlations.get)
            
            tuning = NeuronTuning(
                neuron_idx=neuron_idx,
                pitch_correlation=pitch_corr if not np.isnan(pitch_corr) else 0,
                energy_correlation=energy_corr if not np.isnan(energy_corr) else 0,
                spectral_correlation=spectral_corr if not np.isnan(spectral_corr) else 0,
                primary_feature=primary_feature
            )
            tunings.append(tuning)
        
        # Cluster neurons based on their tuning profiles
        if n_neurons > 3:
            tuning_matrix = np.array([
                [t.pitch_correlation, t.energy_correlation, t.spectral_correlation] 
                for t in tunings
            ])
            
            # Use hierarchical clustering
            n_clusters = min(5, n_neurons // 10)  # Adaptive number of clusters
            if n_clusters > 1:
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(tuning_matrix)
                
                for i, tuning in enumerate(tunings):
                    tuning.cluster_id = cluster_labels[i]
        
        # Sort neurons by cluster and then by correlation strength
        tunings.sort(key=lambda t: (
            t.cluster_id,
            -abs(getattr(t, f"{t.primary_feature}_correlation", 0))
        ))
        
        # Extract sorted indices
        sorted_indices = [t.neuron_idx for t in tunings]
        
        # Compute PCA on the reordered activations
        reordered_activations = activations[:, sorted_indices]

        # Ensure we never ask PCA for more components than available samples/neurons.
        n_time_steps, n_reordered_neurons = reordered_activations.shape
        effective_components = min(self.n_components, n_time_steps, n_reordered_neurons)

        if effective_components > 0:
            if effective_components >= self.n_components:
                pca = PCA(n_components=self.n_components)
                pc_timeseries = pca.fit_transform(reordered_activations)
                explained_variance = pca.explained_variance_ratio_
            else:
                pca = PCA(n_components=effective_components)
                pc_reduced = pca.fit_transform(reordered_activations)
                explained_variance = pca.explained_variance_ratio_

                # Pad PCA outputs so downstream code always receives `self.n_components` tracks.
                pc_timeseries = np.pad(
                    pc_reduced,
                    ((0, 0), (0, self.n_components - effective_components)),
                    mode="constant",
                )
                explained_variance = np.pad(
                    explained_variance,
                    (0, self.n_components - effective_components),
                    mode="constant",
                )
        else:
            pc_timeseries = np.zeros((n_time_steps, self.n_components))
            explained_variance = np.zeros(self.n_components)
        
        # Store results
        self.neuron_order[layer_name] = sorted_indices
        self.neuron_tunings[layer_name] = tunings
        self.pc_timeseries[layer_name] = pc_timeseries
        
        return {
            'sorted_indices': sorted_indices,
            'tunings': tunings,
            'pc_timeseries': pc_timeseries.tolist(),
            'explained_variance': explained_variance.tolist(),
            'reordered_activations': reordered_activations
        }
    
    def compute_layer_stats(self, activations: np.ndarray) -> Dict:
        """Compute rolling statistics for a layer."""
        # Sparsity: proportion of near-zero activations
        threshold = 0.1 * np.max(np.abs(activations))
        sparsity = np.mean(np.abs(activations) < threshold, axis=1)
        
        # Entropy: measure of activation distribution
        # Normalize activations to probabilities
        act_positive = np.abs(activations)
        act_prob = act_positive / (np.sum(act_positive, axis=1, keepdims=True) + 1e-8)
        entropy = -np.sum(act_prob * np.log(act_prob + 1e-8), axis=1)
        
        # Change rate: how fast the representation is changing
        if len(activations) > 1:
            change_rate = np.sqrt(np.sum(np.diff(activations, axis=0)**2, axis=1))
            # Pad to match original length
            change_rate = np.concatenate([[0], change_rate])
        else:
            change_rate = np.zeros(len(activations))
        
        return {
            'sparsity': sparsity.tolist(),
            'entropy': entropy.tolist(),
            'change_rate': change_rate.tolist()
        }
