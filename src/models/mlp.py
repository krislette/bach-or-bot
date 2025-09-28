
"""
MLP Classifier for AI vs Human Music Detection
==============================================

This is our main classifier that determines if a piece of music was created by AI or by humans.

What it does:
- Takes combined features from LLM2Vec (text) + Spectra (audio) 
- Feeds them through a neural network
- Outputs: "This sounds like AI" or "This sounds human"

Quick Start:
---------------------------
# 1. Load settings from config file
config = load_config("config/model_config.yml")

# 2. Combine LLM2Vec and Spectra features
combined_features = np.concatenate([llm2vec_features, spectra_features], axis=1)

# 3. Create classifier
classifier = MLPClassifier(input_dim=combined_features.shape[1], config=config)

# 4. Train it
history = classifier.train(X_train, y_train, X_val, y_val)

# 5. Test it
results = classifier.evaluate(X_test, y_test)

# 6. Use it for new predictions
probabilities, predictions = classifier.predict(new_music_features)

How the Neural Network Works:
-----------------------------
Input → Hidden Layers → Output
  ↓         ↓           ↓
Features  Processing  AI/Human
(LLM2Vec + (Multiple   (0 or 1)
 Spectra)   layers)

The network learns patterns that help distinguish AI-generated music from human music.
"""

from typing import Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class MLPModel(nn.Module):
    """
    The actual neural network that does the AI vs Human classification.

    What happens inside:
    1. Takes the combined LLM2Vec + Spectra features
    2. Passes them through multiple hidden layers (each layer learns different patterns)
    3. Each layer applies: processing → normalization → activation → dropout
    4. Final layer outputs a probability (0-1) where closer to 1 = "more human-like"

    Args:
        input_dim (int): How many features we have total (LLM2Vec size + Spectra size)
        config (Dict): Settings from the YAML file that specify:
            - "hidden_layers": How many neurons in each layer [128, 64, 32]
            - "dropout": How much to randomly "forget" to prevent overfitting [0.3, 0.5, 0.2]
    """
    
    def __init__(self, input_dim: int, config: Dict):
        """
        Build the neural network architecture based on our config file.
        """
        super(MLPModel, self).__init__()
        
        self.hidden_layers = config["hidden_layers"]
        self.dropout_rates = config["dropout"]
        
        # Build layers with batch normalization
        layers = []
        prev_dim = input_dim
        
        # First, normalize the input features (makes training more stable)
        layers.append(nn.BatchNorm1d(input_dim))
        
        # Build hidden layers
        for i, units in enumerate(self.hidden_layers):
            # Main processing layer
            layers.append(nn.Linear(prev_dim, units))
            
            # Normalize outputs (helps with training stability)
            layers.append(nn.BatchNorm1d(units))
            
            # Activation function (allows network to learn complex patterns)
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            
            # Randomly "forget" some connections to prevent overfitting
            dropout_rate = self.dropout_rates[i] if i < len(self.dropout_rates) else 0.5
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = units
        
        # Final output layer: gives us the AI vs Human probability
        layers.append(nn.Linear(prev_dim, 1))
        # Squeezes output between 0 and 1
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
        
        logger.info(f"Built MLP with {len(self.hidden_layers)} hidden layers: {self.hidden_layers}")
    
    def _initialize_weights(self):
        """
        Set up the starting weights for training.
        
        Uses Xavier initialization - a way to set initial weights
        so the network trains better from the start.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input features through the network to get predictions.
        
        Args:
            x: Our combined music features (LLM2Vec + Spectra)

        Returns:
            Probability that the music is human-composed (0 to 1)
        """
        return self.network(x)
    
    def mixup(X, y, alpha=0.2):
        """Apply MixUp augmentation to a batch."""
        if alpha <= 0:
            return X, y, y, 1.0  # no mixing

        lam = np.random.beta(alpha, alpha)
        batch_size = X.size(0)
        index = torch.randperm(batch_size).to(X.device)

        mixed_X = lam * X + (1 - lam) * X[index]
        y_a, y_b = y, y[index]
        return mixed_X, y_a, y_b, lam


    def mixup_loss(criterion, pred, y_a, y_b, lam):
        """Compute MixUp loss."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MLPClassifier:
    """
    The complete music classifier system that wraps everything together.

    This handles all the training, testing, and prediction logic.

    What it manages:
    - The neural network model
    - Training process (with smart features like early stopping)
    - Making predictions on new music
    - Saving/loading trained models
    """
    
    def __init__(self, input_dim: int, config: Dict):
        """
        Set up the complete classification system.

        Args:
            input_dim (int): Total number of features (LLM2Vec + Spectra combined)
            config (Dict): All our settings from the YAML config file

        This creates:
        - The neural network
        - The training optimizer (Adam - good for most cases)
        - Learning rate scheduler (automatically adjusts learning speed)
        - Loss function (measures how wrong our predictions are)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the neural network
        self.model = MLPModel(input_dim, config).to(self.device)
        
        # Optimizer: the algorithm that improves the network during training
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Scheduler: automatically reduces learning rate if we get stuck
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-7
        )
        
        # Loss function: measures how wrong our predictions are
        self.criterion = nn.BCELoss()
        
        self.is_trained = False
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """
        Convert the numpy arrays into batches that PyTorch can process.
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=shuffle)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train the model to recognize AI vs Human music patterns.

        The model learns by:
        1. Looking at training examples (music + labels)
        2. Making predictions
        3. Seeing how wrong it was
        4. Adjusting its parameters to do better
        5. Repeating thousands of times

        Args:
            X_train: Training music features (LLM2Vec + Spectra combined)
            y_train: Training labels (0 = AI-generated, 1 = human-composed)
            X_val: Validation features (used to check if we're overfitting)
            y_val: Validation labels

        Returns:
            Dict: Training history showing how loss and accuracy changed over time

        Smart features included:
        - Early stopping: stops training if validation performance gets worse
        - Learning rate scheduling: slows down learning if we get stuck
        - Gradient clipping: prevents training from going crazy
        - Progress bars: so we can see what's happening. imported tqdm for this LMAO
        """
        logger.info("Starting MLP training...")
        
        # Prepare the data for training
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        # Track training progress
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config["patience"]
        
        # Main training loop
        for epoch in range(self.config["epochs"]):
            # Training phase - model learns from training data
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
            for batch_X, batch_y in train_pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass: make predictions
                self.optimizer.zero_grad()

                # Adding training augmentation if mixup value > 0
                if self.config.get("mixup_alpha", 0) > 0:
                    mixed_X, y_a, y_b, lam = MLPModel.mixup(batch_X, batch_y, alpha=self.config["mixup_alpha"])
                    outputs = self.model(mixed_X)
                    loss = MLPModel.mixup_loss(self.criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                
                # Backward pass: learn from mistakes
                loss.backward()

                # Prevent gradients from getting too large (helps stability)
                if self.config.get("gradient_clipping"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["gradient_clipping"]
                    )

                self.optimizer.step()
                
                # Track statistics
                train_loss += loss.item()
                # Convert probabilities to 0/1 predictions
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*train_correct/train_total:.2f}%'})
            
            # Calculate epoch averages
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase - check how well we do on unseen data
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Adjust learning rate if needed
            self.scheduler.step(val_loss)
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping logic - save best model and stop if no improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best version
                self.save_model("models/mlp/mlp_best.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.is_trained = True
        logger.info("MLP training completed!")
        return history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Test how well the model performs on validation/test data.
        
        This runs the model in "evaluation mode" - no learning happens,
        we just check how accurate our predictions are.
        
        Returns:
            Average loss and accuracy percentage
        """
        # Switch to evaluation mode
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Don't track gradients (saves memory and time)
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
                # Convert to binary predictions
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return avg_val_loss, val_acc
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the trained model to classify new music as AI-generated or human-composed.

        Args:
            X: Music features (LLM2Vec + Spectra combined) for songs we want to classify

        Returns:
            probabilities: How confident the model is (0.0 to 1.0, higher = more human-like)
            predictions: Binary classifications (0 = AI-generated, 1 = human-composed)
        
        Example:
            probs, preds = classifier.predict(new_song_features)
            if preds[0] == 1:
                print(f"This sounds human-composed (confidence: {probs[0]:.2f})")
            else:
                print(f"This sounds AI-generated (confidence: {1-probs[0]:.2f})")
        """
        self.model.eval()
        # Create dummy labels since we don't know the true answers
        data_loader = self._create_data_loader(X, np.zeros(len(X)), shuffle=False)
        
        probabilities = []
        
        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probabilities.extend(outputs.cpu().numpy())
        
        probabilities = np.array(probabilities).flatten()
        # Threshold at 0.5
        predictions = (probabilities > 0.5).astype(int)
        
        return probabilities, predictions
    
    def predict_single(self, features: np.ndarray) -> Tuple[float, int, str]:
        """
        Predict whether a single song is AI-generated or human-composed.
        
        This method is optimized for predicting one song at a time.
        
        Args:
            features: Music features for ONE song (LLM2Vec + Spectra combined)
                    Should be 1D array with shape (feature_dim,)
        
        Returns:
            probability: Confidence score (0.0 to 1.0, higher = more human-like)
            prediction: Binary classification (0 = AI-generated, 1 = human-composed)  
            label: Human-readable label ("AI-Generated" or "Human-Composed")
            
        Example:
            # For a single song
            single_song_features = np.array([0.1, 0.5, 0.3, ...]) 
            prob, pred, label = classifier.predict_single(single_song_features)
            
            print(f"Prediction: {label}")
            print(f"Confidence: {prob:.3f}")
            
            if pred == 1:
                print(f"This sounds {prob:.1%} human-composed")
            else:
                print(f"This sounds {(1-prob):.1%} AI-generated")
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        # Ensure input is the right shape
        if features.ndim == 1:
            features = features.reshape(1, -1)  # Convert to batch of size 1
        elif features.shape[0] != 1:
            raise ValueError(f"Expected features for 1 song, got {features.shape[0]} songs. Use predict_batch() instead.")
        
        # Use the existing predict method
        probabilities, predictions = self.predict(features)
        
        # Extract single results
        probability = float(probabilities[0])
        prediction = int(predictions[0])
        label = "Human-Composed" if prediction == 1 else "AI-Generated"
        
        return probability, prediction, label

    def predict_batch(self, features: np.ndarray, return_details: bool = False) -> Dict:
        """
        Predict AI vs Human classification for multiple songs at once.
        
        This method is optimized for batch processing - much faster than calling
        predict_single() multiple times.
        
        Args:
            features: Music features for MULTIPLE songs (LLM2Vec + Spectra combined)
                    Should be 2D array with shape (num_songs, feature_dim)
            return_details: If True, includes additional statistics and breakdowns
        
        Returns:
            Dictionary containing:
            - 'probabilities': Confidence scores for each song (0.0 to 1.0)
            - 'predictions': Binary classifications (0 = AI, 1 = Human)
            - 'labels': Human-readable labels for each song
            - 'summary': Quick stats about the batch results
            - 'details': (if return_details=True) Additional analysis
            
        Example:
            # For multiple songs
            batch_features = np.array([[0.1, 0.5, 0.3, ...], # Song 1
                                    [0.2, 0.4, 0.7, ...],    # Song 2  
                                    [0.3, 0.6, 0.1, ...]])   # Song 3
                                    
            results = classifier.predict_batch(batch_features, return_details=True)
            
            print(f"Processed {len(results['predictions'])} songs")
            print(f"Summary: {results['summary']}")
            
            for i, (prob, pred, label) in enumerate(zip(results['probabilities'], 
                                                    results['predictions'], 
                                                    results['labels'])):
                print(f"Song {i+1}: {label} (confidence: {prob:.3f})")
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        # Ensure input is 2D
        if features.ndim == 1:
            raise ValueError("For batch prediction, features should be 2D (num_songs, feature_dim). "
                            "For single song, use predict_single() instead.")
        
        num_songs = features.shape[0]
        logger.info(f"Processing batch of {num_songs} songs...")
        
        # Get predictions using existing method
        probabilities, predictions = self.predict(features)
        
        # Convert to human-readable labels
        labels = ["Human-Composed" if pred == 1 else "AI-Generated" for pred in predictions]
        
        # Calculate summary statistics
        num_human = np.sum(predictions == 1)
        num_ai = np.sum(predictions == 0)
        avg_confidence_human = np.mean(probabilities[predictions == 1]) if num_human > 0 else 0.0
        avg_confidence_ai = np.mean(1 - probabilities[predictions == 0]) if num_ai > 0 else 0.0
        
        summary = {
            'total_songs': num_songs,
            'human_composed': num_human,
            'ai_generated': num_ai,
            'human_percentage': (num_human / num_songs) * 100,
            'ai_percentage': (num_ai / num_songs) * 100,
            'avg_confidence_human': avg_confidence_human,
            'avg_confidence_ai': avg_confidence_ai
        }
        
        results = {
            'probabilities': probabilities,
            'predictions': predictions,
            'labels': labels,
            'summary': summary
        }
        
        # Add detailed analysis if requested
        if return_details:
            # Confidence distribution analysis
            high_confidence = np.sum((probabilities > 0.8) | (probabilities < 0.2))
            medium_confidence = np.sum((probabilities >= 0.6) & (probabilities <= 0.8) | 
                                    (probabilities >= 0.2) & (probabilities <= 0.4))
            low_confidence = np.sum((probabilities > 0.4) & (probabilities < 0.6))
            
            # Most confident predictions
            sorted_indices = np.argsort(np.abs(probabilities - 0.5))[::-1]  # Most confident first
            most_confident_indices = sorted_indices[:min(5, len(sorted_indices))]
            least_confident_indices = sorted_indices[-min(5, len(sorted_indices)):]
            
            details = {
                'confidence_distribution': {
                    'high_confidence': high_confidence,
                    'medium_confidence': medium_confidence, 
                    'low_confidence': low_confidence
                },
                'most_confident_predictions': {
                    'indices': most_confident_indices.tolist(),
                    'probabilities': probabilities[most_confident_indices].tolist(),
                    'predictions': predictions[most_confident_indices].tolist()
                },
                'least_confident_predictions': {
                    'indices': least_confident_indices.tolist(), 
                    'probabilities': probabilities[least_confident_indices].tolist(),
                    'predictions': predictions[least_confident_indices].tolist()
                },
                'probability_stats': {
                    'mean': float(np.mean(probabilities)),
                    'std': float(np.std(probabilities)),
                    'min': float(np.min(probabilities)),
                    'max': float(np.max(probabilities)),
                    'median': float(np.median(probabilities))
                }
            }
            results['details'] = details
        
        logger.info(f"Batch prediction completed: {num_human} human, {num_ai} AI-generated")
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Get detailed performance metrics on test data.

        This gives us the final report card for our model:
        - How accurate is it overall?
        - How well does it detect AI-generated music?
        - How well does it detect human-composed music?
        - What kinds of mistakes does it make?

        Args:
            X_test: Test music features
            y_test: True labels (0 = AI, 1 = Human)

        Returns:
            Dictionary with test loss and accuracy
            
        Also logs detailed reports including:
        - Precision, recall, F1-score for each class
        - Confusion matrix showing prediction vs reality
        """
        probabilities, predictions = self.predict(X_test)
        
        test_loader = self._create_data_loader(X_test, y_test, shuffle=False)
        test_loss, test_acc = self._validate(test_loader)
        
        results = {'test_loss': test_loss, 'test_accuracy': test_acc}
        logger.info(f"Test Results: {results}")
        
        # Detailed performance breakdown
        report = classification_report(y_test, predictions, target_names=['AI-Generated', 'Human-Composed'])
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix: shows what the model confused
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save our trained model so we can use it later.

        Args:
            filepath: Where to save the model
        
        Saves everything needed to reload the model:
        - The learned weights
        - Training settings
        - Optimizer state
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model.

        Args:
            filepath: Path to our saved model file
            
        After this, you can immediately use predict() and evaluate()
        without needing to train again.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.is_trained = checkpoint.get('is_trained', True)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> None:
        """
        Print out details about our model architecture.
        
        Useful for debugging or understanding what we've built.
        Shows the network structure and how many parameters it has.
        """
        logger.info("Model Architecture:")
        logger.info(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")


def build_mlp(input_dim: int, config: Dict) -> MLPClassifier:
    """
    Quick way to create an MLP classifier.
    
    Args:
        input_dim: Size of our combined features (LLM2Vec + Spectra)
        config: Our model settings from the YAML file
        
    Returns:
        Ready-to-use MLPClassifier instance
    """
    return MLPClassifier(input_dim, config)


def load_config(config_path: str = "config/model_config.yml") -> Dict:
    """
    Load our model settings from the YAML configuration file.

    Args:
        config_path: Path to our config file

    Returns:
        Dictionary with all our MLP settings (hidden layers, dropout, etc.)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["mlp"]