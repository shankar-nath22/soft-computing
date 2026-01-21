"""Unit tests for Neural Network module."""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network.mlp import MLP


class TestMLP(unittest.TestCase):
    """Test cases for Multi-Layer Perceptron."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randint(0, 2, (100, 1))
        
    def test_initialization(self):
        """Test MLP initialization."""
        mlp = MLP(layers=[5, 10, 1])
        self.assertEqual(len(mlp.weights), 2)
        self.assertEqual(len(mlp.biases), 2)
        self.assertEqual(mlp.weights[0].shape, (5, 10))
        self.assertEqual(mlp.weights[1].shape, (10, 1))
    
    def test_activation_functions(self):
        """Test activation functions."""
        x = np.array([[-1, 0, 1]])
        
        # Sigmoid
        sigmoid_out = MLP.sigmoid(x)
        self.assertTrue(np.all((sigmoid_out >= 0) & (sigmoid_out <= 1)))
        
        # ReLU
        relu_out = MLP.relu(x)
        self.assertTrue(np.all(relu_out >= 0))
        self.assertEqual(relu_out[0, 0], 0)
        self.assertEqual(relu_out[0, 2], 1)
        
        # Tanh
        tanh_out = MLP.tanh(x)
        self.assertTrue(np.all((tanh_out >= -1) & (tanh_out <= 1)))
    
    def test_forward_pass(self):
        """Test forward propagation."""
        mlp = MLP(layers=[5, 10, 1])
        activations, z_values = mlp.forward(self.X)
        
        self.assertEqual(len(activations), 3)
        self.assertEqual(len(z_values), 2)
        self.assertEqual(activations[-1].shape, (100, 1))
    
    def test_training(self):
        """Test training process."""
        mlp = MLP(layers=[5, 10, 1], epochs=10, learning_rate=0.1)
        mlp.train(self.X, self.y, verbose=False)
        
        self.assertEqual(len(mlp.history['loss']), 10)
        self.assertTrue(mlp.history['loss'][-1] < mlp.history['loss'][0])
    
    def test_prediction(self):
        """Test prediction."""
        mlp = MLP(layers=[5, 10, 1], epochs=5)
        mlp.train(self.X, self.y, verbose=False)
        
        predictions = mlp.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
    
    def test_multiclass(self):
        """Test multi-class classification."""
        y_multiclass = np.eye(3)[np.random.randint(0, 3, 100)]
        mlp = MLP(layers=[5, 10, 3], epochs=5)
        mlp.train(self.X, y_multiclass, verbose=False)
        
        predictions = mlp.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(np.all((predictions >= 0) & (predictions < 3)))


if __name__ == '__main__':
    unittest.main()
