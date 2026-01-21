"""Convolutional Neural Network wrapper using TensorFlow/Keras."""

import numpy as np
from typing import Tuple, Optional, List
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN functionality will be limited.")


class CNN:
    """
    Convolutional Neural Network wrapper for image classification.
    
    Uses TensorFlow/Keras for implementation.
    
    Attributes:
        input_shape (Tuple[int, int, int]): Input image shape (height, width, channels)
        num_classes (int): Number of output classes
        model: Keras model instance
        history: Training history
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10,
        architecture: str = 'simple'
    ):
        """
        Initialize CNN.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
            architecture: Model architecture ('simple', 'deep', 'custom')
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN. Install with: pip install tensorflow")
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        self.history = None
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the CNN architecture."""
        if self.architecture == 'simple':
            self.model = self._build_simple_cnn()
        elif self.architecture == 'deep':
            self.model = self._build_deep_cnn()
        else:
            self.model = self._build_simple_cnn()
    
    def _build_simple_cnn(self) -> keras.Model:
        """
        Build a simple CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_deep_cnn(self) -> keras.Model:
        """
        Build a deeper CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the CNN.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0, 1, or 2)
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input images
            
        Returns:
            Predicted class labels
        """
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input images
            
        Returns:
            Prediction probabilities for each class
        """
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X: Input images
            y: True labels (one-hot encoded)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        return self.model.evaluate(X, y, verbose=0)
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
    
    def get_features(self, X: np.ndarray, layer_name: Optional[str] = None) -> np.ndarray:
        """
        Extract features from a specific layer.
        
        Args:
            X: Input images
            layer_name: Name of the layer to extract features from (if None, uses last conv layer)
            
        Returns:
            Features from the specified layer
        """
        if layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(self.model.layers):
                if isinstance(layer, layers.Conv2D):
                    layer_name = layer.name
                    break
        
        feature_extractor = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        return feature_extractor.predict(X, verbose=0)
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
