# Dataset Information

This directory is for storing datasets used in the soft computing examples.

## Supported Datasets

### 1. MNIST
- **Description**: Handwritten digit classification (0-9)
- **Auto-loaded**: Yes (via TensorFlow/Keras)
- **Size**: 60,000 training images, 10,000 test images
- **Image Size**: 28x28 pixels, grayscale
- **Usage**: See `examples/image_classification.py`

### 2. Fashion-MNIST
- **Description**: Fashion item classification (10 categories)
- **Auto-loaded**: Yes (via TensorFlow/Keras)
- **Size**: 60,000 training images, 10,000 test images
- **Image Size**: 28x28 pixels, grayscale
- **Usage**: Can be loaded using `DataLoader.load_fashion_mnist()`

### 3. Synthetic Datasets
The project can generate synthetic datasets for testing:
- **Classification**: Using `DataLoader.generate_synthetic_classification()`
- **Regression**: Using `DataLoader.generate_synthetic_regression()`
- **Time Series**: Using `DataLoader.generate_time_series()`

## Adding Custom Datasets

To use your own datasets:

1. Place your data files in this directory
2. Create a custom loader function in `src/utils/data_loader.py`
3. Use the loader in your examples

Example:
```python
from src.utils.data_loader import DataLoader

# For custom datasets
X_train, y_train, X_test, y_test = load_your_custom_data()
```

## Data Preprocessing

The `DataLoader` class provides utilities for:
- Normalization and standardization
- Train-test splitting
- One-hot encoding for classification
- Image reshaping for CNNs
- Feature extraction

See `src/utils/data_loader.py` for all available functions.
