import logging
import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params

log = logging.getLogger(__name__)

def evaluate_ansatz_expectation(features):
    """
    Evaluate ⟨Z⊗Z⊗...Z⟩ on local AerSimulator (blocking).

    Args:
        features (List[float]): Quantum input features of length num_qubits * layers.

    Returns:
        float: Expectation value.
    """
    num_qubits = 18
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)

    if len(features) != total_params:
        raise ValueError(f"Expected {total_params} features, got {len(features)}")

    # Build parameterized ansatz
    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)
    param_dict = dict(zip(params, features))
    qc = circuit.assign_parameters(param_dict)
    qc.measure_all()

    # Simulate using AerSimulator
    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    # Compute expectation value manually from Z measurements
    expectation = 0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        # Convert most significant bit to 0 or 1
        z_value = 1 if bitstring[::-1][0] == '0' else -1
        expectation += z_value * count

    expectation /= shots
    log.info(f"✅ Local Expectation Value: {expectation:.4f}")
    return expectation


def predict_with_expectation(features, threshold=0.00):
    """
    Predict label using blocking expectation value evaluation.

    Args:
        features (List[float]): Input vector.
        threshold (float): Decision boundary.

    Returns:
        str: Prediction string.
    """
    value = evaluate_ansatz_expectation(features)
    return "Kidney Stone Detected" if value > threshold else "No Kidney Stone"
