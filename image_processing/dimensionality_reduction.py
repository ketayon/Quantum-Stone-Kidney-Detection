import pandas as pd
from sklearn.decomposition import PCA
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from image_processing.data_loader import X_train, X_test

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def reduce_to_n_dimensions(data, n_dimensions):
    """
    Reduces the number of features in the dataset to n dimensions by averaging subsets of features.
    
    Args:
        data (np.ndarray): Input dataset of shape (n_samples, n_features).
        n_dimensions (int): Target number of dimensions.
    
    Returns:
        np.ndarray: Reduced dataset of shape (n_samples, n_dimensions).
    """
    n_features = data.shape[1]
    if n_dimensions > n_features:
        raise ValueError("Target dimensions cannot be greater than the number of features in the dataset.")
    
    # Split features into n_dimensions equal parts
    split_size = n_features // n_dimensions
    reduced_data = np.zeros((data.shape[0], n_dimensions))
    
    for i in range(n_dimensions):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_dimensions - 1 else n_features
        reduced_data[:, i] = np.mean(data[:, start_idx:end_idx], axis=1)
    
    return reduced_data

X_train_red_2d = X_train.reshape(X_train.shape[0], -1)
X_test_red_2d = X_test.reshape(X_test.shape[0], -1)

# Apply Dimensionality Reduction
n_dimensions = 18
X_train_red = reduce_to_n_dimensions(X_train_red_2d, n_dimensions)
X_test_red = reduce_to_n_dimensions(X_test_red_2d, n_dimensions)

# Normalize Data
X_train_reduced = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_train_red)
X_test_reduced = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_test_red)

log.info("Image processing pipeline completed. Data saved in datasets folder.")
