import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_aer import Aer
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params
from quantum_classification.kernel_learning import ansatz, kernel
from quantum_classification.noise_mitigation import apply_noise_mitigation
from quantum_classification.quantum_model import pegasos_svc
from image_processing.dimensionality_reduction import X_test_reduced
from image_processing.data_loader import y_test

# ------------------------------
# Test Quantum Circuit Building
# ------------------------------
def test_calculate_total_params():
    assert calculate_total_params(6) == 18
    assert calculate_total_params(4, layers=2) == 8

def test_build_ansatz_structure():
    num_qubits = 6
    total_params = calculate_total_params(num_qubits)
    params = [Parameter(f"θ{i}") for i in range(total_params)]
    qc = build_ansatz(num_qubits, params)

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == num_qubits
    assert qc.name == "Ansatz"

# ------------------------------
# Test Kernel and PegasosQSVC
# ------------------------------
def test_kernel_is_fidelity():
    assert isinstance(kernel, FidelityQuantumKernel)
    assert kernel.feature_map.name == "Ansatz"

# ------------------------------
# Test PegasosQSVC Model
# ------------------------------
@pytest.mark.parametrize("num_samples", [10])
def test_pegasos_qsvc_predict_shape(num_samples):
    from image_processing.dimensionality_reduction import X_train_reduced
    from image_processing.data_loader import y_train
    pegasos_svc.fit(X_train_reduced[:num_samples], y_train[:num_samples])
    preds = pegasos_svc.predict(X_test_reduced[:num_samples])
    assert len(preds) == num_samples

@pytest.mark.parametrize("num_samples", [10])
def test_pegasos_qsvc_score_range(num_samples):
    from image_processing.dimensionality_reduction import X_train_reduced
    from image_processing.data_loader import y_train
    pegasos_svc.fit(X_train_reduced[:num_samples], y_train[:num_samples])
    score = pegasos_svc.score(X_test_reduced[:num_samples], y_test[:num_samples])
    assert 0.0 <= score <= 1.0

# ------------------------------
# Test Noise Mitigation
# ------------------------------
def test_noise_mitigation_model():
    backend = Aer.get_backend("aer_simulator")
    noise_model = apply_noise_mitigation(backend)
    assert noise_model is not None
    assert hasattr(noise_model, "basic_device_properties") or hasattr(noise_model, "to_dict")

# ------------------------------
# Test Training Wrapper
# ------------------------------
def test_train_and_save_qsvc(tmp_path, monkeypatch):
    from quantum_classification.quantum_model import train_and_save_qsvc
    monkeypatch.setattr("quantum_classification.quantum_model.pegasos_svc.save", lambda path: path)
    accuracy = train_and_save_qsvc()
    assert 0.0 <= accuracy <= 1.0


if __name__ == "__main__":
    pytest.main()
