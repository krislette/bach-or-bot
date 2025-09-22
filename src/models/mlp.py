
"""
MLP (Multi-Layer Perceptron) classifier for binary classification using PyTorch.

This module implements an MLP model for distinguishing between
human-composed and AI-generated music based on extracted features.
Uses LeakyReLU activation and Adam optimizer.
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
    Multi-Layer Perceptron model using PyTorch with LeakyReLU activation.
    """
    
    def __init__(self, input_dim: int, config: Dict):
        super(MLPModel, self).__init__()
        
        self.hidden_layers = config["hidden_layers"]
        self.dropout_rates = config["dropout"]
        
        # Build layers with batch normalization
        layers = []
        prev_dim = input_dim
        
        # Input layer with batch normalization
        layers.append(nn.BatchNorm1d(input_dim))
        
        # Build hidden layers
        for i, units in enumerate(self.hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, units))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(units))
            
            # LeakyReLU activation
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            
            # Dropout layer
            dropout_rate = self.dropout_rates[i] if i < len(self.dropout_rates) else 0.5
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = units
        
        # Output layer (no batch norm or dropout here)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
        
        logger.info(f"Built MLP with {len(self.hidden_layers)} hidden layers: {self.hidden_layers}")
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class MLPClassifier:
    """
    PyTorch-based MLP classifier with LeakyReLU and Adam optimizer.
    """
    
    def __init__(self, input_dim: int, config: Dict):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Dimension of input features
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        self.model = MLPModel(input_dim, config).to(self.device)
        
        # Adam optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.01)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-7
        )
        
        # Binary Cross Entropy loss
        self.criterion = nn.BCELoss()
        
        self.is_trained = False
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=shuffle)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train the MLP model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting MLP training...")
        
        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config["patience"]
        
        # Training loop
        for epoch in range(self.config["epochs"]):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
            for batch_X, batch_y in train_pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()

                if self.config.get("gradient_clipping"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["gradient_clipping"]
                    )

                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*train_correct/train_total:.2f}%'})
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            self.scheduler.step(val_loss)
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("models/fusion/mlp_best.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.is_trained = True
        logger.info("MLP training completed!")
        return history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return avg_val_loss, val_acc
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on input data."""
        self.model.eval()
        data_loader = self._create_data_loader(X, np.zeros(len(X)), shuffle=False)
        
        probabilities = []
        
        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probabilities.extend(outputs.cpu().numpy())
        
        probabilities = np.array(probabilities).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return probabilities, predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        probabilities, predictions = self.predict(X_test)
        
        test_loader = self._create_data_loader(X_test, y_test, shuffle=False)
        test_loss, test_acc = self._validate(test_loader)
        
        results = {'test_loss': test_loss, 'test_accuracy': test_acc}
        logger.info(f"Test Results: {results}")
        
        # Classification report
        report = classification_report(y_test, predictions, target_names=['AI-Generated', 'Human-Composed'])
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.is_trained = checkpoint.get('is_trained', True)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> None:
        """Print model summary."""
        logger.info("Model Architecture:")
        logger.info(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")


def build_mlp(input_dim: int, config: Dict) -> MLPClassifier:
    """Build MLP classifier."""
    return MLPClassifier(input_dim, config)


def load_config(config_path: str = "config/model_config.yml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["mlp"]