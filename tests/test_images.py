import os
import numpy as np
import pytest
from PIL import Image
import warnings
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.utils import algorithm_globals
from image_processing.data_loader import load_images_from_folder, load_and_limit_data
from image_processing.image_transformations import apply_grayscale, apply_gaussian_blur, apply_histogram_equalization
from image_processing.dimensionality_reduction import reduce_to_n_dimensions


algorithm_globals.random_seed = 12345

# Define mock dataset paths for Kidney Stone Classification
test_dataset_path = "./tests/mock_dataset"
mock_kidney_stone_path = os.path.join(test_dataset_path, "stone")
mock_normal_kidney_path = os.path.join(test_dataset_path, "Normal")

# Ensure test dataset directories exist
os.makedirs(mock_kidney_stone_path, exist_ok=True)
os.makedirs(mock_normal_kidney_path, exist_ok=True)

# Create mock ultrasound images
def create_mock_image(file_path):
    """Creates a dummy grayscale ultrasound image for testing"""
    img = Image.new('RGB', (256, 256), color='gray')
    img.save(file_path)

# Create test images
for i in range(5):
    create_mock_image(os.path.join(mock_kidney_stone_path, f"stone_{i}.jpg"))
    create_mock_image(os.path.join(mock_normal_kidney_path, f"normal_{i}.jpg"))


def debug_shape(name, array):
    print(f"{name} shape: {array.shape}")
    if len(array) > 0:
        print(f"{name} first row: {array[0]}")
    else:
        print(f"{name} is empty")

def build_ansatz(num_qubits):
    """Rebuilds the ansatz dynamically to match num_qubits."""
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    params = [Parameter(f"θ{i}") for i in range(num_qubits)]
    
    for i in range(num_qubits):
        ansatz.rx(params[i], i)
    
    return ansatz

@pytest.mark.parametrize("num_qubits, X_train_shape, X_test_shape", [
    (4, (80, 4), (20, 4)),
    (6, (100, 6), (20, 6)),
    (8, (120, 8), (30, 8))
])
def test_pegasos_qsvc(num_qubits, X_train_shape, X_test_shape):
    """Test PegasosQSVC model training and scoring with correct feature mapping."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        # Generate synthetic data with exactly `num_qubits` features
        X_train = np.random.rand(*X_train_shape)
        X_test = np.random.rand(*X_test_shape)
        y_train = np.random.randint(0, 2, X_train_shape[0])
        y_test = np.random.randint(0, 2, X_test_shape[0])
        
        debug_shape("X_train (raw)", X_train)
        debug_shape("X_test (raw)", X_test)
        
        assert X_train.shape[1] == num_qubits, f"X_train shape {X_train.shape} does not match num_qubits {num_qubits}"
        assert X_test.shape[1] == num_qubits, f"X_test shape {X_test.shape} does not match num_qubits {num_qubits}"
        
        # Apply MinMaxScaler to match quantum encoding range [0, π]
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        debug_shape("X_train_scaled", X_train_scaled)
        debug_shape("X_test_scaled", X_test_scaled)
        
        # **Fix: Rebuild ansatz dynamically to match num_qubits**
        dynamic_ansatz = build_ansatz(num_qubits)
        test_kernel = FidelityQuantumKernel(feature_map=dynamic_ansatz)
        
        # Train PegasosQSVC model
        pegasos_svc = PegasosQSVC(quantum_kernel=test_kernel, C=1000, num_steps=100)
        pegasos_svc.fit(X_train_scaled, y_train)
        score = pegasos_svc.score(X_test_scaled, y_test)
        
        print(f"PegasosQSVC Score: {score}")
        
        assert 0 <= score <= 1

# Test Data Loader
@pytest.mark.parametrize("folder, label", [(mock_kidney_stone_path, 1), (mock_normal_kidney_path, 0)])
def test_load_images_from_folder(folder, label):
    """Tests loading kidney ultrasound images from kidney stone & normal folders"""
    data, labels = load_images_from_folder(folder, label)
    assert len(data) == 5
    assert len(labels) == 5
    assert all(lbl == label for lbl in labels)
    assert isinstance(data[0], np.ndarray)  # Ensure numpy array

# Test Data Limiting
@pytest.mark.parametrize("folder, label, num_samples", [(mock_kidney_stone_path, 1, 3), (mock_normal_kidney_path, 0, 2)])
def test_load_and_limit_data(folder, label, num_samples):
    """Tests loading and limiting kidney ultrasound images"""
    data, labels = load_and_limit_data(folder, label, num_samples)
    assert len(data) == num_samples
    assert len(labels) == num_samples
    assert all(lbl == label for lbl in labels)
    assert isinstance(data[0], np.ndarray)  # Ensure numpy array

# Test Image Transformations
def test_apply_grayscale():
    """Tests grayscale conversion on a kidney ultrasound image"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    gray_img = apply_grayscale(img)
    assert len(gray_img.shape) == 2  # Should be single channel
    assert gray_img.dtype == np.uint8  # Check data type

def test_apply_gaussian_blur():
    """Tests Gaussian blur application"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    blurred_img = apply_gaussian_blur(img)
    assert blurred_img.shape == img.shape  # Shape should remain the same
    assert blurred_img.dtype == np.uint8  # Check data type

def test_apply_histogram_equalization():
    """Tests histogram equalization on grayscale kidney ultrasound image"""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    enhanced_img = apply_histogram_equalization(img)
    assert enhanced_img.shape == img.shape  # Shape should remain same
    assert enhanced_img.dtype == np.uint8  # Check data type

# Test Dimensionality Reduction
def test_reduce_to_n_dimensions():
    """Tests dimensionality reduction from 64 to 8"""
    mock_data = np.random.rand(10, 64)  # 10 samples, 64 features
    reduced_data = reduce_to_n_dimensions(mock_data, 8)  # Reduce to 8 dimensions
    assert reduced_data.shape == (10, 8)

if __name__ == "__main__":
    pytest.main()
