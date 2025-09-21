"""
Base LIME implementation for MusicLIME explainability.

This module contains the core LIME algorithm implementation that generates
local explanations for any black-box classifier by perturbing input features.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from typing import Tuple, Callable, Optional, Any
import warnings


class LimeBase:
    """
    Base implementation of Local Interpretable Model-agnostic Explanations (LIME).

    This class implements the core LIME algorithm for generating local explanations
    of black-box classifiers by learning a local linear model around a specific instance.
    """

    def __init__(
        self,
        kernel_fn: Optional[Callable] = None,
        kernel_width: float = 0.25,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the LIME base explainer.

        Parameters
        ----------
        kernel_fn : Optional[Callable], default=None
            Kernel function to weight neighbors by distance. If None, uses exponential kernel
        kernel_width : float, default=0.25
            Width parameter for the kernel function
        verbose : bool, default=False
            Whether to print progress information
        random_state : Optional[int], default=None
            Random seed for reproducibility
        """
        self.kernel_fn = (
            kernel_fn if kernel_fn is not None else self._exponential_kernel
        )
        self.kernel_width = kernel_width
        self.verbose = verbose
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def _exponential_kernel(self, distances: np.ndarray) -> np.ndarray:
        """
        Exponential kernel function for weighting neighbors.

        Parameters
        ----------
        distances : np.ndarray
            Array of distances between perturbed instances and original instance

        Returns
        -------
        np.ndarray
            Kernel weights for each distance
        """
        return np.sqrt(np.exp(-(distances**2) / (self.kernel_width**2)))

    def _generate_neighborhood(
        self, n_features: int, n_samples: int, sample_around_instance: bool = True
    ) -> np.ndarray:
        """
        Generate neighborhood of binary perturbation vectors.

        Parameters
        ----------
        n_features : int
            Number of interpretable features
        n_samples : int
            Number of samples to generate
        sample_around_instance : bool, default=True
            Whether to include the original instance (all features active)

        Returns
        -------
        np.ndarray
            Binary matrix of shape (n_samples, n_features) representing feature presence
        """
        if sample_around_instance:
            # Generate random binary vectors
            neighborhood = np.random.randint(0, 2, (n_samples - 1, n_features))
            # Add the original instance (all features present)
            original_instance = np.ones((1, n_features))
            neighborhood = np.vstack([original_instance, neighborhood])
        else:
            neighborhood = np.random.randint(0, 2, (n_samples, n_features))

        return neighborhood.astype(bool)

    def _compute_distances(
        self,
        neighborhood: np.ndarray,
        original_instance: np.ndarray,
        distance_metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute distances between neighborhood instances and original instance.

        Parameters
        ----------
        neighborhood : np.ndarray
            Binary matrix of perturbed instances
        original_instance : np.ndarray
            Binary vector representing the original instance
        distance_metric : str, default='cosine'
            Distance metric to use ('cosine', 'euclidean', 'manhattan')

        Returns
        -------
        np.ndarray
            Distances from each neighborhood instance to original instance
        """
        # Reshape original instance if needed
        if original_instance.ndim == 1:
            original_instance = original_instance.reshape(1, -1)

        # Compute pairwise distances
        distances = pairwise_distances(
            neighborhood.astype(int),
            original_instance.astype(int),
            metric=distance_metric,
        ).ravel()

        return distances

    def _fit_local_model(
        self,
        neighborhood: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray,
        alpha: float = 1.0,
    ) -> Tuple[Ridge, np.ndarray]:
        """
        Fit a local linear model using weighted Ridge regression.

        Parameters
        ----------
        neighborhood : np.ndarray
            Binary matrix of perturbed instances
        predictions : np.ndarray
            Classifier predictions for each perturbed instance
        weights : np.ndarray
            Sample weights based on kernel function
        alpha : float, default=1.0
            Regularization strength for Ridge regression

        Returns
        -------
        Tuple[Ridge, np.ndarray]
            Fitted Ridge model and feature importance scores
        """
        # Handle binary classification vs regression/multiclass
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            # Multiclass: use probability of predicted class
            predicted_class = np.argmax(predictions, axis=1)
            y = predictions[np.arange(len(predictions)), predicted_class]
        else:
            # Binary classification or regression
            y = predictions.ravel()

        # Fit weighted Ridge regression
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(neighborhood.astype(int), y, sample_weight=weights)

        # Get feature importance (coefficients)
        feature_importance = ridge.coef_

        return ridge, feature_importance

    def explain_instance(
        self,
        instance_id: Any,
        classifier_fn: Callable,
        perturbation_fn: Callable,
        n_features: int,
        n_samples: int = 1000,
        distance_metric: str = "cosine",
        model_regressor: Optional[Any] = None,
        alpha: float = 1.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate LIME explanation for a specific instance.

        Parameters
        ----------
        instance_id : Any
            Identifier for the instance to explain
        classifier_fn : Callable
            Function that takes perturbed data and returns predictions
        perturbation_fn : Callable
            Function that takes binary mask and returns perturbed instance data
        n_features : int
            Number of interpretable features
        n_samples : int, default=1000
            Number of perturbed samples to generate
        distance_metric : str, default='cosine'
            Distance metric for computing neighborhood distances
        model_regressor : Optional[Any], default=None
            Custom regressor to use. If None, uses Ridge regression
        alpha : float, default=1.0
            Regularization strength for the local model

        Returns
        -------
        Tuple[np.ndarray, dict]
            Feature importance scores and explanation metadata

        Raises
        ------
        ValueError
            If n_samples is less than n_features + 1
        RuntimeError
            If explanation generation fails
        """
        if n_samples < n_features + 1:
            raise ValueError(
                f"n_samples ({n_samples}) must be at least n_features + 1 ({n_features + 1})"
            )

        if self.verbose:
            print(f"Generating explanation for instance {instance_id}")
            print(f"Using {n_samples} samples with {n_features} features")

        try:
            # Generate neighborhood of binary perturbations
            neighborhood = self._generate_neighborhood(n_features, n_samples)

            if self.verbose:
                print("Generated neighborhood of perturbations")

            # Get predictions for perturbed instances
            predictions = []
            for i, binary_mask in enumerate(neighborhood):
                if self.verbose and i % (n_samples // 10) == 0:
                    print(f"Processing perturbation {i+1}/{n_samples}")

                # Generate perturbed data using the binary mask
                perturbed_data = perturbation_fn(binary_mask)

                # Get classifier prediction
                pred = classifier_fn(perturbed_data)
                predictions.append(pred)

            predictions = np.array(predictions)

            if self.verbose:
                print("Collected all predictions")

            # Compute distances and weights
            original_instance = np.ones(n_features)  # All features present
            distances = self._compute_distances(
                neighborhood, original_instance, distance_metric
            )
            weights = self.kernel_fn(distances)

            # Fit local linear model
            if model_regressor is None:
                local_model, feature_importance = self._fit_local_model(
                    neighborhood, predictions, weights, alpha
                )
            else:
                # Use custom regressor
                local_model = model_regressor
                local_model.fit(
                    neighborhood.astype(int), predictions.ravel(), sample_weight=weights
                )
                feature_importance = getattr(local_model, "coef_", None)

                if feature_importance is None:
                    warnings.warn(
                        "Custom regressor does not have coef_ attribute. "
                        "Feature importance will be None."
                    )

            # Prepare explanation metadata
            explanation_metadata = {
                "instance_id": instance_id,
                "n_features": n_features,
                "n_samples": n_samples,
                "distance_metric": distance_metric,
                "local_r2_score": (
                    getattr(local_model, "score", lambda x, y: None)(
                        neighborhood.astype(int), predictions.ravel()
                    )
                    if hasattr(local_model, "score")
                    else None
                ),
                "intercept": getattr(local_model, "intercept_", None),
                "kernel_width": self.kernel_width,
            }

            if self.verbose:
                print("Explanation generated successfully")
                if explanation_metadata["local_r2_score"] is not None:
                    print(
                        f"Local model R² score: {explanation_metadata['local_r2_score']:.3f}"
                    )

            return feature_importance, explanation_metadata

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate explanation for instance {instance_id}: {str(e)}"
            )

    def set_random_state(self, random_state: Optional[int]):
        """
        Set random state for reproducibility.

        Parameters
        ----------
        random_state : Optional[int]
            Random seed to set, or None to disable seeding
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def __str__(self) -> str:
        """String representation of the explainer."""
        return f"LimeBase(kernel_width={self.kernel_width}, random_state={self.random_state})"

    def __repr__(self) -> str:
        """Detailed representation of the explainer."""
        return self.__str__()
