"""Prediction and forecasting example using ANFIS."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_systems.neuro_fuzzy import ANFIS
from src.neural_network.mlp import MLP
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualizer


def generate_time_series_data(n_samples: int = 1000) -> tuple:
    """Generate synthetic time series data for prediction."""
    # Generate time series with trend and seasonality
    t = np.arange(n_samples)
    signal = 10 + 0.01 * t  # Trend
    signal += 5 * np.sin(2 * np.pi * t / 100)  # Seasonality
    signal += np.random.normal(0, 0.5, n_samples)  # Noise
    
    return t, signal


def create_sequences(data: np.ndarray, window_size: int = 5) -> tuple:
    """Create input-output sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


def main():
    """Run prediction/forecasting example."""
    print("=" * 60)
    print("Prediction & Forecasting Example - Time Series with ANFIS")
    print("=" * 60)
    
    # Generate time series data
    print("\n1. Generating synthetic time series data...")
    t, signal = generate_time_series_data(n_samples=500)
    print(f"   Time series length: {len(signal)}")
    
    # Create sequences
    print("\n2. Creating input-output sequences...")
    window_size = 5
    X, y = create_sequences(signal, window_size=window_size)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Normalize data
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # Example 1: MLP for comparison
    print("\n3. Training MLP for time series prediction...")
    mlp = MLP(
        layers=[window_size, 10, 1],
        learning_rate=0.01,
        epochs=100,
        activation='tanh'
    )
    mlp.train(X_train_norm, y_train_norm.reshape(-1, 1), verbose=True)
    
    mlp_pred_norm = mlp.predict_proba(X_test_norm).flatten()
    mlp_pred = mlp_pred_norm * y_std + y_mean
    
    mlp_mse = np.mean((mlp_pred - y_test) ** 2)
    print(f"\n   MLP Test MSE: {mlp_mse:.4f}")
    
    # Visualize MLP training
    print("\n4. Visualizing MLP training...")
    Visualizer.plot_training_history(
        mlp.history,
        metrics=['loss'],
        title='MLP Training for Time Series Prediction'
    )
    
    # Example 2: ANFIS
    print("\n5. Training ANFIS for time series prediction...")
    anfis = ANFIS(
        n_inputs=window_size,
        n_mf_per_input=2,
        learning_rate=0.01,
        epochs=100
    )
    anfis.train(X_train_norm, y_train_norm, verbose=True)
    
    anfis_pred_norm = anfis.predict(X_test_norm)
    anfis_pred = anfis_pred_norm * y_std + y_mean
    
    anfis_mse = np.mean((anfis_pred - y_test) ** 2)
    print(f"\n   ANFIS Test MSE: {anfis_mse:.4f}")
    
    # Visualize ANFIS training
    print("\n6. Visualizing ANFIS training...")
    Visualizer.plot_training_history(
        anfis.history,
        metrics=['loss'],
        title='ANFIS Training for Time Series Prediction'
    )
    
    # Compare predictions
    print("\n7. Comparing predictions...")
    test_time = t[train_size + window_size:]
    
    Visualizer.plot_time_series_prediction(
        test_time[:100],  # Show first 100 points for clarity
        y_test[:100],
        mlp_pred[:100],
        title='MLP Time Series Prediction'
    )
    
    Visualizer.plot_time_series_prediction(
        test_time[:100],
        y_test[:100],
        anfis_pred[:100],
        title='ANFIS Time Series Prediction'
    )
    
    # Scatter plots
    print("\n8. Visualizing prediction accuracy...")
    Visualizer.plot_predictions(
        y_test,
        mlp_pred,
        title='MLP: Predicted vs Actual'
    )
    
    Visualizer.plot_predictions(
        y_test,
        anfis_pred,
        title='ANFIS: Predicted vs Actual'
    )
    
    # Performance comparison
    print("\n9. Performance Comparison:")
    print(f"   MLP MSE: {mlp_mse:.4f}")
    print(f"   ANFIS MSE: {anfis_mse:.4f}")
    
    if anfis_mse < mlp_mse:
        improvement = (mlp_mse - anfis_mse) / mlp_mse * 100
        print(f"   ANFIS is {improvement:.2f}% better than MLP")
    else:
        improvement = (anfis_mse - mlp_mse) / anfis_mse * 100
        print(f"   MLP is {improvement:.2f}% better than ANFIS")
    
    print("\n" + "=" * 60)
    print("Prediction & Forecasting Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
