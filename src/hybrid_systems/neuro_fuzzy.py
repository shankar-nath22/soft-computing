"""Adaptive Neuro-Fuzzy Inference System (ANFIS) implementation."""

import numpy as np
from typing import List, Tuple, Optional


class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System.
    
    A hybrid system combining neural networks and fuzzy logic for
    function approximation and prediction tasks.
    
    Architecture:
    - Layer 1: Fuzzification (membership functions)
    - Layer 2: Rules (product of membership degrees)
    - Layer 3: Normalization
    - Layer 4: Consequent parameters
    - Layer 5: Defuzzification (weighted sum)
    
    Attributes:
        n_inputs (int): Number of input features
        n_mf_per_input (int): Number of membership functions per input
        learning_rate (float): Learning rate for training
        epochs (int): Number of training epochs
    """
    
    def __init__(
        self,
        n_inputs: int = 2,
        n_mf_per_input: int = 3,
        learning_rate: float = 0.01,
        epochs: int = 100
    ):
        """
        Initialize ANFIS.
        
        Args:
            n_inputs: Number of input features
            n_mf_per_input: Number of membership functions per input
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
        """
        self.n_inputs = n_inputs
        self.n_mf_per_input = n_mf_per_input
        self.n_rules = n_mf_per_input ** n_inputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Membership function parameters (mean and sigma for Gaussian)
        self.mf_params = []
        for i in range(n_inputs):
            # Initialize uniformly distributed centers and sigmas
            centers = np.linspace(0, 1, n_mf_per_input)
            sigmas = np.ones(n_mf_per_input) * 0.3
            self.mf_params.append({'centers': centers, 'sigmas': sigmas})
        
        # Consequent parameters (linear combination weights)
        self.consequent_params = np.random.randn(self.n_rules, n_inputs + 1) * 0.1
        
        self.history = {'loss': []}
    
    def _gaussian_mf(self, x: float, center: float, sigma: float) -> float:
        """
        Gaussian membership function.
        
        Args:
            x: Input value
            center: Center of Gaussian
            sigma: Width of Gaussian
            
        Returns:
            Membership degree
        """
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def _fuzzify(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Layer 1: Fuzzification.
        
        Args:
            X: Input data of shape (n_samples, n_inputs)
            
        Returns:
            List of membership degrees for each input
        """
        memberships = []
        for i in range(self.n_inputs):
            mf_values = np.zeros((X.shape[0], self.n_mf_per_input))
            for j in range(self.n_mf_per_input):
                center = self.mf_params[i]['centers'][j]
                sigma = self.mf_params[i]['sigmas'][j]
                mf_values[:, j] = self._gaussian_mf(X[:, i], center, sigma)
            memberships.append(mf_values)
        return memberships
    
    def _apply_rules(self, memberships: List[np.ndarray]) -> np.ndarray:
        """
        Layer 2: Apply fuzzy rules (T-norm product).
        
        Args:
            memberships: List of membership degrees
            
        Returns:
            Firing strengths of all rules
        """
        n_samples = memberships[0].shape[0]
        firing_strengths = np.ones((n_samples, self.n_rules))
        
        # Generate all combinations of membership functions
        rule_idx = 0
        if self.n_inputs == 1:
            for i in range(self.n_mf_per_input):
                firing_strengths[:, rule_idx] = memberships[0][:, i]
                rule_idx += 1
        elif self.n_inputs == 2:
            for i in range(self.n_mf_per_input):
                for j in range(self.n_mf_per_input):
                    firing_strengths[:, rule_idx] = (
                        memberships[0][:, i] * memberships[1][:, j]
                    )
                    rule_idx += 1
        else:
            # General case for any number of inputs
            import itertools
            indices = [range(self.n_mf_per_input) for _ in range(self.n_inputs)]
            for combo in itertools.product(*indices):
                strength = np.ones(n_samples)
                for inp_idx, mf_idx in enumerate(combo):
                    strength *= memberships[inp_idx][:, mf_idx]
                firing_strengths[:, rule_idx] = strength
                rule_idx += 1
        
        return firing_strengths
    
    def _normalize(self, firing_strengths: np.ndarray) -> np.ndarray:
        """
        Layer 3: Normalize firing strengths.
        
        Args:
            firing_strengths: Firing strengths from rules
            
        Returns:
            Normalized firing strengths
        """
        sum_strengths = np.sum(firing_strengths, axis=1, keepdims=True)
        # Avoid division by zero
        sum_strengths = np.where(sum_strengths == 0, 1e-10, sum_strengths)
        return firing_strengths / sum_strengths
    
    def _consequent_layer(
        self,
        X: np.ndarray,
        normalized_strengths: np.ndarray
    ) -> np.ndarray:
        """
        Layer 4: Calculate consequent outputs.
        
        Args:
            X: Input data
            normalized_strengths: Normalized firing strengths
            
        Returns:
            Consequent outputs for each rule
        """
        n_samples = X.shape[0]
        consequents = np.zeros((n_samples, self.n_rules))
        
        # Add bias term to input
        X_with_bias = np.column_stack([X, np.ones(n_samples)])
        
        for i in range(self.n_rules):
            # Linear combination: f_i = p_i * x_1 + q_i * x_2 + ... + r_i
            consequents[:, i] = np.dot(X_with_bias, self.consequent_params[i])
        
        return consequents
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through ANFIS.
        
        Args:
            X: Input data of shape (n_samples, n_inputs)
            
        Returns:
            Tuple of (output, layer_outputs)
        """
        # Layer 1: Fuzzification
        memberships = self._fuzzify(X)
        
        # Layer 2: Rules
        firing_strengths = self._apply_rules(memberships)
        
        # Layer 3: Normalization
        normalized_strengths = self._normalize(firing_strengths)
        
        # Layer 4: Consequent layer
        consequents = self._consequent_layer(X, normalized_strengths)
        
        # Layer 5: Defuzzification (weighted sum)
        output = np.sum(normalized_strengths * consequents, axis=1)
        
        layer_outputs = {
            'memberships': memberships,
            'firing_strengths': firing_strengths,
            'normalized_strengths': normalized_strengths,
            'consequents': consequents
        }
        
        return output, layer_outputs
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train ANFIS using hybrid learning (gradient descent + least squares).
        
        Args:
            X: Training data of shape (n_samples, n_inputs)
            y: Target values of shape (n_samples,)
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            verbose: Whether to print progress
        """
        for epoch in range(self.epochs):
            # Forward pass
            predictions, layer_outputs = self.forward(X)
            
            # Calculate loss
            loss = np.mean((predictions - y) ** 2)
            self.history['loss'].append(loss)
            
            # Backward pass - update consequent parameters using least squares
            normalized_strengths = layer_outputs['normalized_strengths']
            X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
            
            for i in range(self.n_rules):
                # Weighted input for this rule
                weighted_X = X_with_bias * normalized_strengths[:, i:i+1]
                
                # Least squares update
                try:
                    self.consequent_params[i] = np.linalg.lstsq(
                        weighted_X, y * normalized_strengths[:, i], rcond=None
                    )[0]
                except (np.linalg.LinAlgError, ValueError):
                    # If singular, use gradient descent
                    error = predictions - y
                    gradient = np.dot(
                        weighted_X.T,
                        error * normalized_strengths[:, i]
                    ) / X.shape[0]
                    self.consequent_params[i] -= self.learning_rate * gradient
            
            # Update membership function parameters using gradient descent
            error = predictions - y
            for i in range(self.n_inputs):
                for j in range(self.n_mf_per_input):
                    # Calculate gradients
                    center = self.mf_params[i]['centers'][j]
                    sigma = self.mf_params[i]['sigmas'][j]
                    
                    # Gradient w.r.t center and sigma (simplified)
                    mf_values = self._gaussian_mf(X[:, i], center, sigma)
                    
                    d_center = np.mean(error * mf_values * (X[:, i] - center) / (sigma ** 2))
                    d_sigma = np.mean(error * mf_values * ((X[:, i] - center) ** 2) / (sigma ** 3))
                    
                    # Update parameters
                    self.mf_params[i]['centers'][j] -= self.learning_rate * d_center
                    self.mf_params[i]['sigmas'][j] -= self.learning_rate * d_sigma
                    
                    # Keep sigma positive
                    self.mf_params[i]['sigmas'][j] = max(0.01, self.mf_params[i]['sigmas'][j])
            
            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                val_msg = ""
                if X_val is not None and y_val is not None:
                    val_pred, _ = self.forward(X_val)
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    val_msg = f", Val Loss: {val_loss:.4f}"
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}{val_msg}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        predictions, _ = self.forward(X)
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model performance (MSE).
        
        Args:
            X: Input data
            y: True values
            
        Returns:
            Mean squared error
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
