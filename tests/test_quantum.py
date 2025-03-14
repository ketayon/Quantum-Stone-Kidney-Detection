import pytest
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from quantum_classification.quantum_circuit import build_ansatz, conv_layer, pool_layer, calculate_total_params
from quantum_classification.kernel_learning import kernel
from quantum_classification.noise_mitigation import apply_noise_mitigation

def test_conv_layer():
    """Test convolutional layer with RX gates."""
    num_qubits = 4
    params = [Parameter(f'θ{i}') for i in range(num_qubits)]
    conv = conv_layer(num_qubits, "conv_test", params)
    assert isinstance(conv, QuantumCircuit)
    assert conv.num_qubits == num_qubits

def test_pool_layer():
    """Test pooling layer with CX gates."""
    num_qubits = 4
    qubits = list(range(num_qubits))
    pool = pool_layer(qubits, "pool_test")
    assert isinstance(pool, QuantumCircuit)
    assert pool.num_qubits == num_qubits

def test_calculate_total_params():
    """Test calculation of total parameters for the ansatz."""
    num_qubits = 6
    expected_params = num_qubits * 3
    assert calculate_total_params(num_qubits) == expected_params

def test_build_ansatz():
    """Test ansatz building with correct structure."""
    num_qubits = 6
    total_params = calculate_total_params(num_qubits)
    params = [Parameter(f'θ{i}') for i in range(total_params)]
    ansatz = build_ansatz(num_qubits, params)
    assert isinstance(ansatz, QuantumCircuit)
    assert ansatz.num_qubits == num_qubits

def test_pegasos_qsvc():
    """Test PegasosQSVC model initialization."""
    pegasos_svc = PegasosQSVC(quantum_kernel=kernel, C=1000, num_steps=100)
    assert isinstance(pegasos_svc, PegasosQSVC)

def test_noise_mitigation():
    """Test noise mitigation application."""
    backend_mock = AerSimulator()  # Proper Aer backend for noise model testing
    noise_model = apply_noise_mitigation(backend_mock)
    assert isinstance(noise_model, NoiseModel)

def test_kernel_initialization():
    """Test Fidelity Quantum Kernel initialization."""
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    test_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=build_ansatz(6, [Parameter(f'θ{i}') for i in range(18)]))
    assert isinstance(test_kernel, FidelityQuantumKernel)

if __name__ == "__main__":
    pytest.main()