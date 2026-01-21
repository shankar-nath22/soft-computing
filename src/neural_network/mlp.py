"""Multi-Layer Perceptron implementation from scratch using NumPy."""

import numpy as np
from typing import List, Callable, Tuple, Optional


class MLP:
    """
    Multi-Layer Perceptron (MLP) Neural Network.
    
    A feedforward neural network with backpropagation training.
    
    Attributes:
        layers (List[int]): List of layer sizes including input and output layers
        learning_rate (float): Learning rate for gradient descent
        epochs (int): Number of training epochs
        activation (str): Activation function name ('sigmoid', 'relu', 'tanh')
        weights (List[np.ndarray]): List of weight matrices
        biases (List[np.ndarray]): List of bias vectors
        history (dict): Training history
    """
    
    def __init__(
        self,
        layers: List[int],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        activation: str = 'sigmoid'
    ):
        """
        Initialize MLP.
        
        Args:
            layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            activation: Activation function ('sigmoid', 'relu', 'tanh', 'softmax')
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        self.weights = []
        self.biases = []
        self.history = {'loss': [], 'accuracy': []}
        
        # Initialize weights and biases
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize weights using Xavier initialization and biases to zeros."""
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (self.layers[i] + self.layers[i + 1]))
            weight = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i + 1]))
            bias = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = MLP.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _get_activation(self) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        activations = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'relu': (self.relu, self.relu_derivative),
            'tanh': (self.tanh, self.tanh_derivative)
        }
        return activations.get(self.activation_name, (self.sigmoid, self.sigmoid_derivative))
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple of (activations, z_values) for each layer
        """
        activation_func, _ = self._get_activation()
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Use softmax for output layer in multi-class classification
            if i == len(self.weights) - 1 and self.layers[-1] > 1:
                a = self.softmax(z)
            else:
                a = activation_func(z)
            activations.append(a)
        
        return activations, z_values
    
    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray]
    ):
        """
        Backward propagation.
        
        Args:
            X: Input data
            y: True labels
            activations: Activations from forward pass
            z_values: Pre-activation values from forward pass
        """
        _, activation_derivative = self._get_activation()
        m = X.shape[0]
        
        # Calculate output layer error
        if self.layers[-1] > 1:  # Multi-class classification
            delta = activations[-1] - y
        else:  # Binary classification or regression
            delta = (activations[-1] - y) * activation_derivative(z_values[-1])
        
        # Backpropagate error
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * activation_derivative(z_values[i - 1])
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train the MLP.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print progress
        """
        # Ensure y is 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        for epoch in range(self.epochs):
            # Forward pass
            activations, z_values = self.forward(X)
            
            # Backward pass
            self.backward(X, y, activations, z_values)
            
            # Calculate loss
            predictions = activations[-1]
            if self.layers[-1] > 1:  # Multi-class
                loss = -np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1))
            else:  # Binary or regression
                loss = np.mean((predictions - y) ** 2)
            
            self.history['loss'].append(loss)
            
            # Calculate accuracy for classification
            if self.layers[-1] > 1:
                acc = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            else:
                acc = np.mean((predictions > 0.5) == y) if np.all(np.isin(y, [0, 1])) else 0
            self.history['accuracy'].append(acc)
            
            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                val_msg = ""
                if X_val is not None and y_val is not None:
                    val_acc = self.evaluate(X_val, y_val)
                    val_msg = f", Val Acc: {val_acc:.4f}"
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}, Acc: {acc:.4f}{val_msg}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        activations, _ = self.forward(X)
        predictions = activations[-1]
        
        if self.layers[-1] > 1:  # Multi-class
            return np.argmax(predictions, axis=1)
        else:  # Binary or regression
            return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Prediction probabilities
        """
        activations, _ = self.forward(X)
        return activations[-1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y.flatten()
        return np.mean(predictions == y_true)
