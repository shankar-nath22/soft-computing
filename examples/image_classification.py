"""Image classification example using hybrid neuro-fuzzy approach."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network.mlp import MLP
from src.neural_network.cnn import CNN
from src.hybrid_systems.ga_nn_fuzzy import GANeuralFuzzy
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualizer


def main():
    """Run image classification example on MNIST."""
    print("=" * 60)
    print("Image Classification Example - MNIST Dataset")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MNIST dataset...")
    try:
        X_train, y_train, X_test, y_test = DataLoader.load_mnist(subset_size=5000)
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
    except Exception as e:
        print(f"   Error loading MNIST: {e}")
        print("   Generating synthetic data instead...")
        X_train, y_train, X_test, y_test = DataLoader.generate_synthetic_classification(
            n_samples=1000, n_features=784, n_classes=10
        )
    
    # Prepare data for MLP
    print("\n2. Preparing data for MLP...")
    X_train_flat, y_train_onehot, X_test_flat, y_test_onehot = DataLoader.prepare_for_mlp(
        X_train, y_train, X_test, y_test, num_classes=10
    )
    
    # Example 1: Simple MLP
    print("\n3. Training simple MLP...")
    mlp = MLP(
        layers=[X_train_flat.shape[1], 128, 64, 10],
        learning_rate=0.01,
        epochs=50,
        activation='relu'
    )
    mlp.train(X_train_flat, y_train_onehot, X_test_flat, y_test_onehot, verbose=True)
    
    mlp_accuracy = mlp.evaluate(X_test_flat, y_test_onehot)
    print(f"\n   MLP Test Accuracy: {mlp_accuracy:.4f}")
    
    # Visualize training history
    print("\n4. Visualizing training history...")
    Visualizer.plot_training_history(
        mlp.history,
        metrics=['loss', 'accuracy'],
        title='MLP Training History'
    )
    
    # Get predictions
    predictions = mlp.predict(X_test_flat[:10])
    print(f"\n5. Sample predictions: {predictions}")
    print(f"   True labels: {y_test[:10]}")
    
    # Visualize confusion matrix
    print("\n6. Generating confusion matrix...")
    all_predictions = mlp.predict(X_test_flat)
    Visualizer.plot_confusion_matrix(
        y_test,
        all_predictions,
        class_names=[str(i) for i in range(10)],
        title='MLP Confusion Matrix'
    )
    
    # Visualize sample images
    print("\n7. Visualizing sample predictions...")
    Visualizer.plot_sample_images(
        X_test[:10],
        y_test[:10],
        all_predictions[:10],
        class_names=[str(i) for i in range(10)],
        n_samples=10,
        title='Sample Predictions'
    )
    
    # Example 2: Hybrid GA-Neural-Fuzzy System (optional, more advanced)
    print("\n8. Training Hybrid GA-Neural-Fuzzy System (simplified)...")
    try:
        hybrid = GANeuralFuzzy(
            network_type='mlp',
            input_shape=None,
            n_classes=10,
            use_fuzzy_output=True
        )
        
        # Train without GA optimization for speed
        hybrid.train_neural_network(
            X_train_flat[:1000],  # Use subset for speed
            y_train_onehot[:1000],
            X_test_flat[:200],
            y_test_onehot[:200],
            hyperparameters={
                'layers': [X_train_flat.shape[1], 64, 32, 10],
                'learning_rate': 0.01,
                'epochs': 20,
                'activation': 'relu'
            },
            verbose=True
        )
        
        # Get predictions with confidence
        hybrid_predictions, confidences = hybrid.predict_with_confidence(X_test_flat[:10])
        print(f"\n   Hybrid predictions: {hybrid_predictions}")
        print(f"   Confidence scores: {confidences}")
        
        hybrid_accuracy = hybrid.evaluate(X_test_flat, y_test_onehot)
        print(f"\n   Hybrid Test Accuracy: {hybrid_accuracy:.4f}")
        
    except Exception as e:
        print(f"   Hybrid system training skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Image Classification Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
