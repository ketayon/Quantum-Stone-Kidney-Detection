import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from image_processing.data_loader import X_train, X_test


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["X_train_reduced", "X_test_reduced"]

def reduce_to_n_dimensions(data, n_dimensions):
    n_features = data.shape[1]
    if n_dimensions > n_features:
        raise ValueError("Target dimensions > input features.")

    split_size = n_features // n_dimensions
    reduced = np.zeros((data.shape[0], n_dimensions))

    for i in range(n_dimensions):
        start, end = i * split_size, (i + 1) * split_size if i < n_dimensions - 1 else n_features
        reduced[:, i] = np.mean(data[:, start:end], axis=1)
    return reduced


# Flatten before reduction
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

use_pca = False
n_dimensions = 18


if use_pca:
    pca = PCA(n_components=n_dimensions)
    X_train_red = pca.fit_transform(X_train_flat)
    X_test_red = pca.transform(X_test_flat)
else:
    X_train_red = reduce_to_n_dimensions(X_train_flat, n_dimensions)
    X_test_red = reduce_to_n_dimensions(X_test_flat, n_dimensions)

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_reduced = scaler.fit_transform(X_train_red)
X_test_reduced = scaler.transform(X_test_red)


log.info("Dimensionality reduction & normalization complete.")
