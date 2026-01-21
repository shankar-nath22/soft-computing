"""Pattern recognition example using GA for feature selection."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network.mlp import MLP
from src.genetic_algorithm.ga_core import GeneticAlgorithm
from src.fuzzy_logic.fuzzy_system import FuzzySystem, FuzzyVariable, FuzzyRule
from src.fuzzy_logic.membership_functions import FuzzySet
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualizer


class GAFeatureSelection:
    """Feature selection using Genetic Algorithm."""
    
    def __init__(
        self,
        n_features: int,
        population_size: int = 50,
        generations: int = 20
    ):
        """
        Initialize GA feature selector.
        
        Args:
            n_features: Total number of features
            population_size: GA population size
            generations: Number of generations
        """
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.ga = None
        self.best_features = None
    
    def fitness_function(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Create fitness function for feature selection."""
        def evaluate(chromosome):
            # Decode chromosome to feature mask
            selected_features = chromosome > 0.5
            n_selected = np.sum(selected_features)
            
            # Require at least one feature
            if n_selected == 0:
                return 0.0
            
            # Select features
            X_train_selected = X_train[:, selected_features]
            X_val_selected = X_val[:, selected_features]
            
            # Train simple classifier
            try:
                classifier = MLP(
                    layers=[n_selected, 32, 10],
                    learning_rate=0.01,
                    epochs=20,
                    activation='relu'
                )
                classifier.train(X_train_selected, y_train, verbose=False)
                
                # Evaluate
                accuracy = classifier.evaluate(X_val_selected, y_val)
                
                # Penalty for too many features
                feature_penalty = n_selected / self.n_features * 0.1
                
                return accuracy - feature_penalty
            except Exception as e:
                # Return 0 fitness if training fails
                return 0.0
        
        return evaluate
    
    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ):
        """
        Select best features using GA.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            verbose: Whether to print progress
        """
        # Create fitness function
        fitness_fn = self.fitness_function(X_train, y_train, X_val, y_val)
        
        # Initialize GA
        self.ga = GeneticAlgorithm(
            population_size=self.population_size,
            chromosome_length=self.n_features,
            generations=self.generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism_count=2,
            fitness_function=fitness_fn,
            bounds=(0, 1),
            encoding='real'
        )
        
        # Run GA
        self.ga.evolve(verbose=verbose)
        
        # Get best features
        best_chromosome, best_fitness = self.ga.get_best_solution()
        self.best_features = best_chromosome > 0.5
        
        if verbose:
            print(f"\nFeature selection complete:")
            print(f"  Selected {np.sum(self.best_features)}/{self.n_features} features")
            print(f"  Best fitness: {best_fitness:.4f}")
        
        return self.best_features


def main():
    """Run pattern recognition example."""
    print("=" * 60)
    print("Pattern Recognition Example - GA Feature Selection")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic classification data...")
    X_train, y_train, X_test, y_test = DataLoader.generate_synthetic_classification(
        n_samples=1000,
        n_features=30,
        n_classes=10,
        test_size=0.2
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    # Prepare labels for MLP
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    
    # Split training data for validation
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train_onehot[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train_onehot[:-val_size]
    
    # Example 1: Baseline classifier with all features
    print("\n2. Training baseline classifier (all features)...")
    baseline = MLP(
        layers=[X_train.shape[1], 64, 32, 10],
        learning_rate=0.01,
        epochs=50,
        activation='relu'
    )
    baseline.train(X_train_sub, y_train_sub, X_val, y_val, verbose=True)
    
    baseline_accuracy = baseline.evaluate(X_test, y_test_onehot)
    print(f"\n   Baseline Test Accuracy: {baseline_accuracy:.4f}")
    
    # Example 2: GA-based feature selection
    print("\n3. Performing GA-based feature selection...")
    feature_selector = GAFeatureSelection(
        n_features=X_train.shape[1],
        population_size=30,
        generations=15
    )
    
    selected_features = feature_selector.select_features(
        X_train_sub, y_train_sub, X_val, y_val, verbose=True
    )
    
    # Visualize GA evolution
    print("\n4. Visualizing feature selection optimization...")
    Visualizer.plot_ga_fitness_evolution(
        feature_selector.ga.history,
        title="GA Feature Selection Evolution"
    )
    
    # Train classifier with selected features
    print("\n5. Training classifier with selected features...")
    X_train_selected = X_train_sub[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    optimized = MLP(
        layers=[np.sum(selected_features), 64, 32, 10],
        learning_rate=0.01,
        epochs=50,
        activation='relu'
    )
    optimized.train(X_train_selected, y_train_sub, X_val_selected, y_val, verbose=True)
    
    optimized_accuracy = optimized.evaluate(X_test_selected, y_test_onehot)
    print(f"\n   Optimized Test Accuracy: {optimized_accuracy:.4f}")
    
    # Compare training histories
    print("\n6. Comparing training histories...")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(baseline.history['loss'], label='All Features', linewidth=2)
    plt.plot(optimized.history['loss'], label='Selected Features', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(baseline.history['accuracy'], label='All Features', linewidth=2)
    plt.plot(optimized.history['accuracy'], label='Selected Features', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrices
    print("\n7. Generating confusion matrices...")
    baseline_pred = baseline.predict(X_test)
    optimized_pred = optimized.predict(X_test_selected)
    
    Visualizer.plot_confusion_matrix(
        y_test, baseline_pred,
        class_names=[str(i) for i in range(10)],
        title='Baseline Classifier Confusion Matrix'
    )
    
    Visualizer.plot_confusion_matrix(
        y_test, optimized_pred,
        class_names=[str(i) for i in range(10)],
        title='Optimized Classifier Confusion Matrix'
    )
    
    # Performance summary
    print("\n8. Performance Summary:")
    print(f"   Baseline (all {X_train.shape[1]} features):")
    print(f"     Test Accuracy: {baseline_accuracy:.4f}")
    
    print(f"\n   Optimized ({np.sum(selected_features)} features):")
    print(f"     Test Accuracy: {optimized_accuracy:.4f}")
    print(f"     Feature Reduction: {(1 - np.sum(selected_features)/X_train.shape[1])*100:.1f}%")
    
    if optimized_accuracy > baseline_accuracy:
        print(f"     Accuracy Improvement: {(optimized_accuracy - baseline_accuracy)*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Pattern Recognition Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
