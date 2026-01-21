"""Data loading utilities for soft computing examples."""

import numpy as np
from typing import Tuple, Optional
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DataLoader:
    """
    Data loading and preprocessing utilities.
    
    Provides methods to load and prepare data for soft computing applications.
    """
    
    @staticmethod
    def load_mnist(subset_size: Optional[int] = None) -> Tuple:
        """
        Load MNIST dataset.
        
        Args:
            subset_size: If specified, load only a subset of data
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        try:
            from tensorflow import keras
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            
            # Normalize
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            
            # Subset if requested
            if subset_size is not None:
                X_train = X_train[:subset_size]
                y_train = y_train[:subset_size]
                X_test = X_test[:min(subset_size // 5, len(X_test))]
                y_test = y_test[:min(subset_size // 5, len(y_test))]
            
            return X_train, y_train, X_test, y_test
        except ImportError:
            raise ImportError("TensorFlow is required to load MNIST. Install with: pip install tensorflow")
    
    @staticmethod
    def load_fashion_mnist(subset_size: Optional[int] = None) -> Tuple:
        """
        Load Fashion-MNIST dataset.
        
        Args:
            subset_size: If specified, load only a subset of data
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        try:
            from tensorflow import keras
            (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
            
            # Normalize
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            
            # Subset if requested
            if subset_size is not None:
                X_train = X_train[:subset_size]
                y_train = y_train[:subset_size]
                X_test = X_test[:min(subset_size // 5, len(X_test))]
                y_test = y_test[:min(subset_size // 5, len(y_test))]
            
            return X_train, y_train, X_test, y_test
        except ImportError:
            raise ImportError("TensorFlow is required to load Fashion-MNIST. Install with: pip install tensorflow")
    
    @staticmethod
    def prepare_for_mlp(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: Optional[int] = None
    ) -> Tuple:
        """
        Prepare image data for MLP.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_test: Test images
            y_test: Test labels
            num_classes: Number of classes for one-hot encoding
            
        Returns:
            Tuple of (X_train_flat, y_train_onehot, X_test_flat, y_test_onehot)
        """
        # Flatten images
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # One-hot encode labels if num_classes is provided
        if num_classes is not None:
            y_train_onehot = np.eye(num_classes)[y_train]
            y_test_onehot = np.eye(num_classes)[y_test]
        else:
            y_train_onehot = y_train
            y_test_onehot = y_test
        
        return X_train_flat, y_train_onehot, X_test_flat, y_test_onehot
    
    @staticmethod
    def prepare_for_cnn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: Optional[int] = None
    ) -> Tuple:
        """
        Prepare image data for CNN.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_test: Test images
            y_test: Test labels
            num_classes: Number of classes for one-hot encoding
            
        Returns:
            Tuple of (X_train, y_train_onehot, X_test, y_test_onehot)
        """
        # Add channel dimension if needed
        if len(X_train.shape) == 3:
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
        
        # One-hot encode labels if num_classes is provided
        if num_classes is not None:
            y_train_onehot = np.eye(num_classes)[y_train]
            y_test_onehot = np.eye(num_classes)[y_test]
        else:
            y_train_onehot = y_train
            y_test_onehot = y_test
        
        return X_train, y_train_onehot, X_test, y_test_onehot
    
    @staticmethod
    def generate_synthetic_classification(
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 2,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple:
        """
        Generate synthetic classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=n_features // 4,
            n_classes=n_classes,
            random_state=random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def generate_synthetic_regression(
        n_samples: int = 1000,
        n_features: int = 10,
        test_size: float = 0.2,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple:
        """
        Generate synthetic regression dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            test_size: Proportion of test set
            noise: Standard deviation of Gaussian noise
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise * 10,
            random_state=random_state
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def generate_time_series(
        n_samples: int = 1000,
        frequency: float = 0.1,
        noise: float = 0.1,
        trend: float = 0.001
    ) -> np.ndarray:
        """
        Generate synthetic time series data.
        
        Args:
            n_samples: Number of time steps
            frequency: Frequency of sine wave
            noise: Amount of random noise
            trend: Linear trend coefficient
            
        Returns:
            Time series array
        """
        t = np.arange(n_samples)
        signal = np.sin(2 * np.pi * frequency * t)
        signal += trend * t
        signal += noise * np.random.randn(n_samples)
        return signal
