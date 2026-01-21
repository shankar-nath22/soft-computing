# Soft Computing Project: Neural Networks, Fuzzy Logic, and Genetic Algorithms

A comprehensive Python-based soft computing library demonstrating the integration of Neural Networks, Fuzzy Logic, and Genetic Algorithms across multiple application domains.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Project Overview

This project provides a complete implementation of soft computing techniques including:
- **Neural Networks** (MLP and CNN)
- **Fuzzy Logic Systems** (Mamdani inference with multiple membership functions)
- **Genetic Algorithms** (with various selection, crossover, and mutation operators)
- **Hybrid Systems** (ANFIS, GA-Fuzzy, and combined neuro-fuzzy-genetic systems)

### Motivation

Soft computing techniques are essential for handling uncertainty, imprecision, and optimization in real-world problems. This project demonstrates how these techniques can be:
- Used individually for specific tasks
- Combined to create more powerful hybrid systems
- Applied to diverse domains (classification, control, forecasting, optimization)

## 📚 Techniques Explained

### Neural Networks
Neural networks are computational models inspired by biological neural networks. They excel at:
- Pattern recognition
- Function approximation
- Feature extraction
- Classification and regression

**Implementation**: We provide both a from-scratch MLP using NumPy and a CNN wrapper using TensorFlow/Keras.

### Fuzzy Logic
Fuzzy logic handles uncertainty and imprecision using linguistic variables and fuzzy rules. It's ideal for:
- Control systems
- Decision making under uncertainty
- Expert systems
- Modeling human reasoning

**Implementation**: Mamdani-type fuzzy inference system with triangular, trapezoidal, and Gaussian membership functions.

### Genetic Algorithms
Genetic algorithms are optimization techniques inspired by natural evolution. They're effective for:
- Hyperparameter optimization
- Feature selection
- Multi-objective optimization
- Search in complex spaces

**Implementation**: Complete GA with tournament/roulette/rank selection, multiple crossover and mutation operators, and elitism.

### Hybrid Systems

#### ANFIS (Adaptive Neuro-Fuzzy Inference System)
Combines neural network learning with fuzzy logic reasoning for adaptive systems.

#### GA-Fuzzy
Uses genetic algorithms to optimize fuzzy system parameters for improved performance.

#### GA-Neural-Fuzzy
Comprehensive system using all three techniques for complex classification tasks.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/shankar-nath22/soft-computing.git
cd soft-computing

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies
- numpy: Numerical computations
- matplotlib: Visualization
- scikit-learn: Data preprocessing and metrics
- tensorflow: CNN implementation
- pandas: Data handling
- seaborn: Enhanced visualization
- scikit-fuzzy: Additional fuzzy logic support

## 🏗️ Project Structure

```
soft-computing/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── src/                         # Source code
│   ├── neural_network/          # Neural network implementations
│   │   ├── mlp.py              # Multi-Layer Perceptron (from scratch)
│   │   └── cnn.py              # CNN wrapper (TensorFlow/Keras)
│   ├── fuzzy_logic/            # Fuzzy logic system
│   │   ├── fuzzy_system.py     # Mamdani fuzzy inference
│   │   └── membership_functions.py  # Membership functions
│   ├── genetic_algorithm/      # Genetic algorithm
│   │   ├── ga_core.py          # Core GA implementation
│   │   └── operators.py        # Selection, crossover, mutation
│   ├── hybrid_systems/         # Hybrid approaches
│   │   ├── neuro_fuzzy.py      # ANFIS implementation
│   │   ├── ga_fuzzy.py         # GA-optimized fuzzy system
│   │   └── ga_nn_fuzzy.py      # Combined system
│   └── utils/                  # Utility functions
│       ├── data_loader.py      # Data loading and preprocessing
│       └── visualization.py    # Plotting and visualization
├── examples/                    # Example applications
│   ├── image_classification.py # MNIST classification
│   ├── control_system.py       # Temperature control
│   ├── prediction_forecasting.py  # Time series with ANFIS
│   └── pattern_recognition.py  # GA feature selection
├── tests/                       # Unit tests
│   ├── test_neural_network.py
│   ├── test_fuzzy_logic.py
│   └── test_genetic_algorithm.py
└── data/                        # Dataset directory
    └── README.md               # Dataset information
```

## 🎯 Quick Start Guide

### Example 1: Neural Network Classification

```python
from src.neural_network.mlp import MLP
from src.utils.data_loader import DataLoader

# Load data
X_train, y_train, X_test, y_test = DataLoader.load_mnist(subset_size=5000)
X_train_flat, y_train_onehot, X_test_flat, y_test_onehot = DataLoader.prepare_for_mlp(
    X_train, y_train, X_test, y_test, num_classes=10
)

# Create and train MLP
mlp = MLP(layers=[784, 128, 64, 10], learning_rate=0.01, epochs=50)
mlp.train(X_train_flat, y_train_onehot)

# Evaluate
accuracy = mlp.evaluate(X_test_flat, y_test_onehot)
print(f"Accuracy: {accuracy:.4f}")
```

### Example 2: Fuzzy Logic Control System

```python
from src.fuzzy_logic.fuzzy_system import FuzzySystem, FuzzyVariable, FuzzyRule
from src.fuzzy_logic.membership_functions import FuzzySet

# Create fuzzy system
fs = FuzzySystem("Temperature Controller")

# Add variables and membership functions
error_var = FuzzyVariable("error", (-20, 20))
error_var.add_fuzzy_set(FuzzySet("negative", "triangular", (-20, -20, 0)))
error_var.add_fuzzy_set(FuzzySet("zero", "triangular", (-5, 0, 5)))
error_var.add_fuzzy_set(FuzzySet("positive", "triangular", (0, 20, 20)))
fs.add_input_variable(error_var)

# Add output variable
power_var = FuzzyVariable("power", (0, 100))
power_var.add_fuzzy_set(FuzzySet("low", "triangular", (0, 0, 50)))
power_var.add_fuzzy_set(FuzzySet("medium", "triangular", (25, 50, 75)))
power_var.add_fuzzy_set(FuzzySet("high", "triangular", (50, 100, 100)))
fs.add_output_variable(power_var)

# Add rules
fs.add_rule(FuzzyRule([("error", "positive")], ("power", "high")))

# Inference
output = fs.inference({"error": 10.0})
print(f"Power: {output['power']:.2f}%")
```

### Example 3: Genetic Algorithm Optimization

```python
from src.genetic_algorithm.ga_core import GeneticAlgorithm

# Define fitness function
def fitness_fn(chromosome):
    return -np.sum((chromosome - 0.5) ** 2)  # Maximize closeness to 0.5

# Create and run GA
ga = GeneticAlgorithm(
    population_size=50,
    chromosome_length=10,
    generations=100,
    fitness_function=fitness_fn
)

ga.evolve(verbose=True)
best_solution, best_fitness = ga.get_best_solution()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

### Example 4: Hybrid ANFIS System

```python
from src.hybrid_systems.neuro_fuzzy import ANFIS
import numpy as np

# Generate training data
X = np.random.rand(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

# Create and train ANFIS
anfis = ANFIS(n_inputs=2, n_mf_per_input=3, epochs=100)
anfis.train(X, y)

# Predict
predictions = anfis.predict(X)
```

## 📖 Usage Examples

### Running the Examples

Each example can be run as a standalone script:

```bash
# Image classification with neural networks
python examples/image_classification.py

# Fuzzy logic control system
python examples/control_system.py

# Time series forecasting with ANFIS
python examples/prediction_forecasting.py

# Pattern recognition with GA feature selection
python examples/pattern_recognition.py
```

### Example Outputs

#### Image Classification
- Training accuracy curves
- Confusion matrices
- Sample predictions with confidence scores
- Comparison of MLP vs hybrid approaches

#### Control System
- Membership function visualizations
- Control response plots
- GA optimization curves
- Performance comparison (manual vs optimized)

#### Prediction/Forecasting
- Time series predictions
- MSE comparisons
- Prediction vs actual plots

#### Pattern Recognition
- Feature selection results
- Classification accuracy improvements
- Confusion matrices

## 🧪 Testing

Run the unit tests to verify the implementation:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test module
python -m unittest tests.test_neural_network
python -m unittest tests.test_fuzzy_logic
python -m unittest tests.test_genetic_algorithm
```

## 📊 Results and Visualizations

The project includes comprehensive visualization capabilities:

### Neural Networks
- Training loss and accuracy curves
- Confusion matrices
- Sample image predictions
- Feature maps (for CNNs)

### Fuzzy Logic
- Membership function plots
- Inference surface plots
- Control response curves

### Genetic Algorithms
- Fitness evolution over generations
- Best/average/worst fitness tracking
- Convergence analysis

### Hybrid Systems
- Combined performance metrics
- Comparative analysis
- Optimization progress

## 🔮 Future Improvements

Potential enhancements for the project:

1. **Additional Techniques**
   - Particle Swarm Optimization (PSO)
   - Ant Colony Optimization (ACO)
   - Deep Q-Learning

2. **More Applications**
   - Stock price prediction
   - Medical diagnosis systems
   - Autonomous navigation
   - Game AI

3. **Performance Optimization**
   - GPU acceleration
   - Parallel processing
   - Caching mechanisms

4. **Interactive Tools**
   - Jupyter notebook tutorials
   - Web-based visualization dashboard
   - Parameter tuning GUI

5. **Advanced Features**
   - Transfer learning support
   - Ensemble methods
   - AutoML capabilities

## 📚 References and Resources

### Books
1. "Neural Networks and Deep Learning" by Michael Nielsen
2. "Fuzzy Logic with Engineering Applications" by Timothy J. Ross
3. "An Introduction to Genetic Algorithms" by Melanie Mitchell

### Papers
1. Jang, J-S. R. (1993). "ANFIS: Adaptive-Network-Based Fuzzy Inference System"
2. Holland, J. H. (1992). "Genetic Algorithms"
3. Zadeh, L. A. (1965). "Fuzzy Sets"

### Online Resources
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Fuzzy Logic Tutorial](https://pythonhosted.org/scikit-fuzzy/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Soft Computing Team

## 🙏 Acknowledgments

- Thanks to the open-source community for the excellent libraries
- Inspired by various soft computing research papers and implementations
- Built for educational purposes and practical applications

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes and demonstrates the integration of soft computing techniques. For production use, consider additional optimizations and validation.
