"""Hybrid system combining Neural Networks, Fuzzy Logic, and Genetic Algorithms."""

import numpy as np
from typing import Callable, Optional, Tuple, List
import sys
sys.path.append('..')
from src.neural_network.mlp import MLP
from src.neural_network.cnn import CNN
from src.fuzzy_logic.fuzzy_system import FuzzySystem, FuzzyVariable, FuzzyRule
from src.fuzzy_logic.membership_functions import FuzzySet
from src.genetic_algorithm.ga_core import GeneticAlgorithm


class GANeuralFuzzy:
    """
    Comprehensive hybrid system using Neural Networks, Fuzzy Logic, and Genetic Algorithms.
    
    Architecture:
    1. Neural Network (CNN/MLP) for feature extraction/learning
    2. Fuzzy Logic for decision making and uncertainty handling
    3. Genetic Algorithm for hyperparameter optimization
    
    Use cases:
    - Image classification with fuzzy decision boundaries
    - Pattern recognition with uncertainty
    - Optimized neural-fuzzy architectures
    
    Attributes:
        network_type (str): Type of neural network ('mlp' or 'cnn')
        neural_net: Neural network instance
        fuzzy_system (FuzzySystem): Fuzzy inference system
        ga (GeneticAlgorithm): Genetic algorithm optimizer
    """
    
    def __init__(
        self,
        network_type: str = 'mlp',
        input_shape: Optional[Tuple] = None,
        n_classes: int = 10,
        use_fuzzy_output: bool = True
    ):
        """
        Initialize hybrid system.
        
        Args:
            network_type: Type of neural network ('mlp' or 'cnn')
            input_shape: Shape of input data
            n_classes: Number of output classes
            use_fuzzy_output: Whether to use fuzzy logic for final classification
        """
        self.network_type = network_type
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.use_fuzzy_output = use_fuzzy_output
        
        self.neural_net = None
        self.fuzzy_system = None
        self.ga = None
        self.best_hyperparameters = None
    
    def _build_neural_network(self, hyperparameters: dict):
        """
        Build neural network with given hyperparameters.
        
        Args:
            hyperparameters: Dictionary of network hyperparameters
        """
        if self.network_type == 'mlp':
            # Build MLP
            layers = hyperparameters.get('layers', [128, 64, self.n_classes])
            learning_rate = hyperparameters.get('learning_rate', 0.01)
            epochs = hyperparameters.get('epochs', 50)
            activation = hyperparameters.get('activation', 'relu')
            
            self.neural_net = MLP(
                layers=layers,
                learning_rate=learning_rate,
                epochs=epochs,
                activation=activation
            )
        
        elif self.network_type == 'cnn':
            # Build CNN
            architecture = hyperparameters.get('architecture', 'simple')
            
            self.neural_net = CNN(
                input_shape=self.input_shape,
                num_classes=self.n_classes,
                architecture=architecture
            )
    
    def _build_fuzzy_classifier(self):
        """Build fuzzy system for classification based on network outputs."""
        self.fuzzy_system = FuzzySystem("Neural-Fuzzy Classifier")
        
        # Create fuzzy variables for class probabilities
        for i in range(min(3, self.n_classes)):  # Limit to 3 for simplicity
            var = FuzzyVariable(f'class_{i}_prob', (0, 1))
            
            # Add membership functions
            var.add_fuzzy_set(FuzzySet('low', 'triangular', (0, 0, 0.5)))
            var.add_fuzzy_set(FuzzySet('medium', 'triangular', (0.2, 0.5, 0.8)))
            var.add_fuzzy_set(FuzzySet('high', 'triangular', (0.5, 1, 1)))
            
            self.fuzzy_system.add_input_variable(var)
        
        # Output: confidence
        output_var = FuzzyVariable('confidence', (0, 1))
        output_var.add_fuzzy_set(FuzzySet('low', 'triangular', (0, 0, 0.5)))
        output_var.add_fuzzy_set(FuzzySet('medium', 'triangular', (0.2, 0.5, 0.8)))
        output_var.add_fuzzy_set(FuzzySet('high', 'triangular', (0.5, 1, 1)))
        self.fuzzy_system.add_output_variable(output_var)
        
        # Add simple rules
        # If class probability is high, confidence is high
        for i in range(min(3, self.n_classes)):
            rule = FuzzyRule(
                antecedents=[(f'class_{i}_prob', 'high')],
                consequent=('confidence', 'high')
            )
            self.fuzzy_system.add_rule(rule)
            
            rule = FuzzyRule(
                antecedents=[(f'class_{i}_prob', 'medium')],
                consequent=('confidence', 'medium')
            )
            self.fuzzy_system.add_rule(rule)
    
    def train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        hyperparameters: Optional[dict] = None,
        verbose: bool = True
    ):
        """
        Train the neural network component.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            hyperparameters: Network hyperparameters
            verbose: Whether to print progress
        """
        if hyperparameters is None:
            if self.network_type == 'mlp':
                hyperparameters = {
                    'layers': [X_train.shape[1], 128, 64, self.n_classes],
                    'learning_rate': 0.01,
                    'epochs': 50,
                    'activation': 'relu'
                }
            else:
                hyperparameters = {
                    'architecture': 'simple'
                }
        
        # Build network
        self._build_neural_network(hyperparameters)
        
        # Train
        if self.network_type == 'mlp':
            self.neural_net.train(X_train, y_train, X_val, y_val, verbose=verbose)
        else:
            epochs = hyperparameters.get('epochs', 10)
            batch_size = hyperparameters.get('batch_size', 32)
            self.neural_net.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size,
                verbose=1 if verbose else 0
            )
        
        # Build fuzzy classifier if enabled
        if self.use_fuzzy_output:
            self._build_fuzzy_classifier()
    
    def predict(self, X: np.ndarray, use_fuzzy: bool = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            use_fuzzy: Whether to use fuzzy logic (default: self.use_fuzzy_output)
            
        Returns:
            Predicted class labels
        """
        if self.neural_net is None:
            raise ValueError("Neural network not trained. Call train_neural_network() first.")
        
        if use_fuzzy is None:
            use_fuzzy = self.use_fuzzy_output
        
        # Get neural network predictions
        predictions = self.neural_net.predict(X)
        
        if use_fuzzy and self.fuzzy_system is not None:
            # Get probabilities
            probabilities = self.neural_net.predict_proba(X)
            
            # Apply fuzzy logic for confidence estimation
            fuzzy_predictions = []
            for prob in probabilities:
                # Use top probabilities for fuzzy inference
                inputs = {}
                for i in range(min(3, self.n_classes)):
                    inputs[f'class_{i}_prob'] = float(prob[i] if i < len(prob) else 0)
                
                fuzzy_output = self.fuzzy_system.inference(inputs)
                fuzzy_predictions.append(fuzzy_output.get('confidence', 0.5))
            
            # Could use fuzzy confidence to adjust predictions
            # For now, return neural network predictions
            return predictions
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if self.neural_net is None:
            raise ValueError("Neural network not trained.")
        
        predictions = self.neural_net.predict(X)
        probabilities = self.neural_net.predict_proba(X)
        
        if self.use_fuzzy_output and self.fuzzy_system is not None:
            # Calculate fuzzy confidence
            confidences = []
            for prob in probabilities:
                inputs = {}
                for i in range(min(3, self.n_classes)):
                    inputs[f'class_{i}_prob'] = float(prob[i] if i < len(prob) else 0)
                
                fuzzy_output = self.fuzzy_system.inference(inputs)
                confidences.append(fuzzy_output.get('confidence', 0.5))
            
            return predictions, np.array(confidences)
        else:
            # Use max probability as confidence
            confidences = np.max(probabilities, axis=1)
            return predictions, confidences
    
    def optimize_with_ga(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_ranges: dict,
        population_size: int = 20,
        generations: int = 10,
        verbose: bool = True
    ):
        """
        Optimize neural network hyperparameters using GA.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            param_ranges: Dictionary of parameter ranges to optimize
            population_size: GA population size
            generations: Number of GA generations
            verbose: Whether to print progress
        """
        def fitness_function(chromosome):
            """Evaluate hyperparameters."""
            # Decode chromosome to hyperparameters
            hyperparams = {}
            idx = 0
            
            for param_name, (min_val, max_val, param_type) in param_ranges.items():
                if param_type == 'int':
                    value = int(min_val + chromosome[idx] * (max_val - min_val))
                else:
                    value = min_val + chromosome[idx] * (max_val - min_val)
                hyperparams[param_name] = value
                idx += 1
            
            # Build and train network
            try:
                if self.network_type == 'mlp':
                    # For MLP, optimize learning rate and hidden layer size
                    test_net = MLP(
                        layers=[X_train.shape[1],
                               hyperparams.get('hidden_size', 64),
                               self.n_classes],
                        learning_rate=hyperparams.get('learning_rate', 0.01),
                        epochs=30,  # Reduced for GA speed
                        activation='relu'
                    )
                    test_net.train(X_train, y_train, verbose=False)
                else:
                    # For CNN, use simple architecture
                    test_net = CNN(
                        input_shape=self.input_shape,
                        num_classes=self.n_classes,
                        architecture='simple'
                    )
                    test_net.train(X_train, y_train, epochs=5, verbose=0)
                
                # Evaluate on validation set
                accuracy = test_net.evaluate(X_val, y_val)
                return accuracy
            except Exception as e:
                # Return 0 fitness if training/evaluation fails
                return 0.0
        
        # Setup GA
        chromosome_length = len(param_ranges)
        
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            chromosome_length=chromosome_length,
            generations=generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism_count=2,
            fitness_function=fitness_function,
            bounds=(0, 1),
            encoding='real'
        )
        
        # Run optimization
        print("Optimizing hyperparameters with GA...")
        self.ga.evolve(verbose=verbose)
        
        # Get best hyperparameters
        best_chromosome, best_fitness = self.ga.get_best_solution()
        
        # Decode best hyperparameters
        self.best_hyperparameters = {}
        idx = 0
        for param_name, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'int':
                value = int(min_val + best_chromosome[idx] * (max_val - min_val))
            else:
                value = min_val + best_chromosome[idx] * (max_val - min_val)
            self.best_hyperparameters[param_name] = value
            idx += 1
        
        if verbose:
            print(f"\nBest hyperparameters found (Fitness: {best_fitness:.4f}):")
            for param, value in self.best_hyperparameters.items():
                print(f"  {param}: {value}")
        
        # Train final model with best hyperparameters
        print("\nTraining final model with optimized hyperparameters...")
        self.train_neural_network(
            X_train, y_train, X_val, y_val,
            hyperparameters=self.best_hyperparameters,
            verbose=verbose
        )
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Accuracy score
        """
        if self.neural_net is None:
            raise ValueError("Neural network not trained.")
        
        return self.neural_net.evaluate(X, y)
