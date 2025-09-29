"""
True MusicLIME implementation matching the original paper.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from typing import List, Callable, Optional, Dict, Any
import warnings

from .lime.base import LimeBase


class IndexedLyrics:
    """Simple lyrics indexer for MusicLIME."""
    
    def __init__(self, lyrics_lines: List[str]):
        self.lyrics_lines = [line.strip() for line in lyrics_lines if line.strip()]
        
    def num_words(self) -> int:
        return len(self.lyrics_lines)
    
    def inverse_removing(self, inactive_indices: List[int]) -> List[str]:
        """Return lyrics with inactive lines removed."""
        result = []
        for i, line in enumerate(self.lyrics_lines):
            if i not in inactive_indices:
                result.append(line)
        return result


class TrueMusicLIMEExplainer:
    """
    True MusicLIME implementation matching the original paper.
    
    This implementation follows the exact architecture from the original
    MusicLIME notebook, ensuring compatibility and performance.
    """
    
    def __init__(
        self,
        kernel_width: float = 25,
        kernel: Optional[Callable] = None,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        kernel_width = float(kernel_width)
        
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        
        self.random_state = np.random.RandomState(random_state)
        self.base = LimeBase(kernel_fn, verbose)
        
    def explain_instance(
        self,
        factorization,  # Audio factorization object
        lyrics_lines: List[str],
        predict_fn: Callable,
        num_features: int = 100000,
        num_samples: int = 1000,
        batch_size: int = 10,
        distance_metric: str = 'cosine',
        modality: str = 'both'
    ):
        """
        Generate MusicLIME explanation.
        
        Parameters
        ----------
        factorization : Factorization
            Audio factorization object
        lyrics_lines : List[str]
            List of lyrics lines
        predict_fn : Callable
            Function that takes (lyrics_lines, audio_array) and returns prediction
        num_features : int
            Maximum number of features to use
        num_samples : int
            Number of perturbation samples
        batch_size : int
            Batch size for predictions
        distance_metric : str
            Distance metric for LIME
        modality : str
            'both', 'audio', or 'lyrics'
        """
        
        if modality not in ['both', 'lyrical', 'audio']:
            raise ValueError('Set modality to "both", "lyrical" or "audio".')
        
        self.factorization = factorization
        
        # Create indexed lyrics
        indexed_lyrics = IndexedLyrics(lyrics_lines)
        
        # Generate perturbations and get predictions
        data, labels, distances = self.combined_data_labels_distances(
            indexed_lyrics, predict_fn, num_samples, 
            batch_size=batch_size, distance_metric=distance_metric, 
            modality=modality
        )
        
        # Create explanation object
        explanation = TrueMusicLIMEExplanation(
            indexed_lyrics, self.factorization, data, labels, modality=modality
        )
        
        # Get top label (assuming binary classification)
        top_label = 0
        
        # Generate LIME explanation
        (explanation.intercept[top_label],
         explanation.local_exp[top_label],
         explanation.score[top_label], 
         explanation.local_pred[top_label]) = self.base.explain_instance_with_data(
            data, labels, distances, top_label, num_features
        )
        
        return explanation
    
    def combined_data_labels_distances(
        self,
        indexed_lyrics: IndexedLyrics,
        predict_fn: Callable,
        num_samples: int,
        modality: str,
        batch_size: int = 10,
        distance_metric: str = 'cosine'
    ):
        """
        Core MusicLIME method that generates perturbations for both modalities.
        
        This is the key method that was missing from your implementation.
        """
        
        doc_size = indexed_lyrics.num_words()
        audio_size = self.factorization.get_number_components()
        
        if modality == 'both':
            total_features = doc_size + audio_size
        elif modality == 'lyrical':
            total_features = doc_size
        elif modality == 'audio':
            total_features = audio_size
        
        # Generate binary perturbation matrix
        data = self.random_state.randint(0, 2, num_samples * total_features)\
            .reshape((num_samples, total_features))
        data[0, :] = 1  # First sample is original (all features active)
        
        labels = []
        audios = []
        texts = []
        
        for row in data:
            if modality == 'both':
                # Split perturbation between audio and lyrics
                audio_mask = row[:audio_size]
                lyrics_mask = row[audio_size:]
                
                # Get active audio components
                active_audio = np.where(audio_mask != 0)[0]
                perturbed_audio = self.factorization.compose_model_input(active_audio)
                audios.append(perturbed_audio)
                
                # Get active lyrics lines
                inactive_lyrics = np.where(lyrics_mask == 0)[0]
                perturbed_lyrics = indexed_lyrics.inverse_removing(inactive_lyrics)
                texts.append(perturbed_lyrics)
                
            elif modality == 'audio':
                active_audio = np.where(row != 0)[0]
                perturbed_audio = self.factorization.compose_model_input(active_audio)
                audios.append(perturbed_audio)
                
                # Use all lyrics
                perturbed_lyrics = indexed_lyrics.inverse_removing([])
                texts.append(perturbed_lyrics)
                
            elif modality == 'lyrical':
                # Use all audio
                all_audio = list(range(audio_size))
                perturbed_audio = self.factorization.compose_model_input(all_audio)
                audios.append(perturbed_audio)
                
                # Perturb lyrics
                inactive_lyrics = np.where(row == 0)[0]
                perturbed_lyrics = indexed_lyrics.inverse_removing(inactive_lyrics)
                texts.append(perturbed_lyrics)
            
            # Process in batches
            if len(audios) == batch_size:
                preds = predict_fn(texts, np.array(audios))
                labels.extend(preds)
                audios = []
                texts = []
        
        # Process remaining batch
        if len(audios) > 0:
            preds = predict_fn(texts, np.array(audios))
            labels.extend(preds)
        
        # Compute distances
        distances = pairwise_distances(
            data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel()
        
        return data, np.array(labels), distances


class TrueMusicLIMEExplanation:
    """Explanation object matching original MusicLIME."""
    
    def __init__(self, indexed_lyrics, factorization, neighborhood_data, 
                 neighborhood_labels, modality):
        self.indexed_lyrics = indexed_lyrics
        self.factorization = factorization
        self.neighborhood_data = neighborhood_data
        self.neighborhood_labels = neighborhood_labels
        self.modality = modality
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}
    
    def get_sorted_components(self, label, positive_components=True, 
                            negative_components=True, num_components='all',
                            min_abs_weight=0.0, return_indices=False):
        """Get sorted components by importance."""
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
        n_audio_features = self.factorization.get_number_components()
        exp = self.local_exp[label]
        
        # Extract feature indices and weights
        w = [[x[0], x[1]] for x in exp]
        used_features, weights = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1]
        
        # Filter by positive/negative
        if not negative_components:
            pos_weights = np.argwhere(weights > 0)[:, 0]
            used_features = used_features[pos_weights]
            weights = weights[pos_weights]
        elif not positive_components:
            neg_weights = np.argwhere(weights < 0)[:, 0]
            used_features = used_features[neg_weights]
            weights = weights[neg_weights]
        
        # Filter by minimum weight
        if min_abs_weight != 0.0:
            abs_weights = np.argwhere(abs(weights) >= min_abs_weight)[:, 0]
            used_features = used_features[abs_weights]
            weights = weights[abs_weights]
        
        # Limit number of components
        if num_components == 'all':
            num_components = len(used_features)
        else:
            assert isinstance(num_components, int)
        
        used_features = used_features[:num_components]
        weights = weights[:num_components]
        
        # Map indices to component names
        components = []
        for index in used_features:
            if self.modality == 'both':
                if index < n_audio_features:
                    components.append(self.factorization.get_ordered_component_names()[index])
                else:
                    lyrics_idx = index - n_audio_features
                    if lyrics_idx < len(self.indexed_lyrics.lyrics_lines):
                        components.append(f"lyrics: {self.indexed_lyrics.lyrics_lines[lyrics_idx]}")
                    else:
                        components.append(f"lyrics_line_{lyrics_idx}")
            elif self.modality == 'lyrical':
                if index < len(self.indexed_lyrics.lyrics_lines):
                    components.append(f"lyrics: {self.indexed_lyrics.lyrics_lines[index]}")
                else:
                    components.append(f"lyrics_line_{index}")
            elif self.modality == 'audio':
                components.append(self.factorization.get_ordered_component_names()[index])
        
        if return_indices:
            return components, weights, used_features
        return components, weights
