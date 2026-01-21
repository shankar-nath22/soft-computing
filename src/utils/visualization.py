"""Visualization utilities for soft computing."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


class Visualizer:
    """
    Visualization utilities for soft computing applications.
    
    Provides methods to visualize neural networks, fuzzy logic, and genetic algorithms.
    """
    
    @staticmethod
    def plot_training_history(
        history: dict,
        metrics: List[str] = ['loss', 'accuracy'],
        title: str = 'Training History',
        save_path: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            history: Dictionary with training history
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(history[metric], label=f'Training {metric}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} vs Epoch')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else range(len(cm)),
            yticklabels=class_names if class_names else range(len(cm))
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_membership_functions(
        fuzzy_variable,
        x_range: Optional[Tuple[float, float]] = None,
        title: str = 'Membership Functions',
        save_path: Optional[str] = None
    ):
        """
        Plot fuzzy membership functions.
        
        Args:
            fuzzy_variable: FuzzyVariable object
            x_range: Range for x-axis (optional)
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        if x_range is None:
            x_range = fuzzy_variable.range
        
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        plt.figure(figsize=(10, 6))
        for name, fuzzy_set in fuzzy_variable.fuzzy_sets.items():
            y = fuzzy_set.membership(x)
            plt.plot(x, y, label=name, linewidth=2)
        
        plt.xlabel(fuzzy_variable.name)
        plt.ylabel('Membership Degree')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_ga_fitness_evolution(
        history: dict,
        title: str = 'Genetic Algorithm Fitness Evolution',
        save_path: Optional[str] = None
    ):
        """
        Plot genetic algorithm fitness evolution.
        
        Args:
            history: GA history dictionary
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(10, 6))
        
        generations = range(len(history['best_fitness']))
        plt.plot(generations, history['best_fitness'], label='Best Fitness', linewidth=2)
        plt.plot(generations, history['avg_fitness'], label='Average Fitness', linewidth=2)
        
        if 'worst_fitness' in history:
            plt.plot(generations, history['worst_fitness'], label='Worst Fitness', linewidth=2, alpha=0.5)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Predictions vs Actual',
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs actual values for regression.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_time_series_prediction(
        time: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = 'Time Series Prediction',
        save_path: Optional[str] = None
    ):
        """
        Plot time series predictions.
        
        Args:
            time: Time steps
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(time, actual, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(time, predicted, label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sample_images(
        images: np.ndarray,
        labels: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        n_samples: int = 10,
        title: str = 'Sample Images',
        save_path: Optional[str] = None
    ):
        """
        Plot sample images with labels and predictions.
        
        Args:
            images: Image array
            labels: True labels
            predictions: Predicted labels (optional)
            class_names: Names of classes
            n_samples: Number of samples to display
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        n_samples = min(n_samples, len(images))
        fig, axes = plt.subplots(2, n_samples // 2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(n_samples):
            ax = axes[i]
            
            # Handle different image shapes
            if len(images[i].shape) == 3 and images[i].shape[-1] == 1:
                ax.imshow(images[i].squeeze(), cmap='gray')
            elif len(images[i].shape) == 2:
                ax.imshow(images[i], cmap='gray')
            else:
                ax.imshow(images[i])
            
            # Create label
            label_text = str(class_names[labels[i]] if class_names else labels[i])
            if predictions is not None:
                pred_text = str(class_names[predictions[i]] if class_names else predictions[i])
                color = 'green' if labels[i] == predictions[i] else 'red'
                label_text = f'True: {label_text}\nPred: {pred_text}'
                ax.set_title(label_text, color=color, fontsize=9)
            else:
                ax.set_title(f'Label: {label_text}', fontsize=9)
            
            ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_control_response(
        time: np.ndarray,
        setpoint: np.ndarray,
        output: np.ndarray,
        control_signal: Optional[np.ndarray] = None,
        title: str = 'Control System Response',
        save_path: Optional[str] = None
    ):
        """
        Plot control system response.
        
        Args:
            time: Time steps
            setpoint: Desired setpoint
            output: Actual output
            control_signal: Control signal (optional)
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        if control_signal is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        
        ax1.plot(time, setpoint, 'r--', label='Setpoint', linewidth=2)
        ax1.plot(time, output, 'b-', label='Output', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title('System Response')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if control_signal is not None:
            ax2.plot(time, control_signal, 'g-', linewidth=2)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Control Signal')
            ax2.set_title('Control Signal')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
