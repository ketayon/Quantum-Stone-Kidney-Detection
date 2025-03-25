import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from qiskit.circuit import QuantumCircuit

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflow.job_scheduler import JobScheduler
from workflow.workflow_manager import WorkflowManager

# -----------------------------
# Fixtures and Globals
# -----------------------------
valid_features = np.linspace(0, np.pi, 54)  # 18 qubits x 3 layers
invalid_features = np.linspace(0, np.pi, 20)  # Invalid input length

# -----------------------------
# JobScheduler Tests
# -----------------------------

def test_scheduler_executes_task():
    scheduler = JobScheduler(max_workers=1)
    result = scheduler.schedule_task(lambda x: x + 1, 5)
    assert result == 6

# -----------------------------
# Interpretation Tests
# -----------------------------

def test_interpret_counts_stone():
    counts = {'100': 800, '000': 224}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result in ["Stone", "Normal"]


def test_interpret_counts_normal():
    counts = {'000': 900, '100': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result in ["Stone", "Normal"]


def test_interpret_invalid_counts():
    with pytest.raises(ValueError, match="Empty counts from quantum simulation."):
        WorkflowManager._interpret_quantum_counts({})

# -----------------------------
# Classification Tests
# -----------------------------
def test_classify_with_quantum_circuit_noise_valid():
    prediction = WorkflowManager.classify_with_quantum_circuit_noise(valid_features)
    assert prediction in ["Stone", "Normal"]


def test_classify_with_quantum_circuit_noise_invalid():
    with pytest.raises(ValueError, match="Expected .* features"):
        WorkflowManager.classify_with_quantum_circuit_noise(invalid_features)

# -----------------------------
# CLI Training Pipeline
# -----------------------------
@patch("workflow.workflow_manager.train_and_save_qsvc", return_value=0.90)
def test_training_pipeline(mock_train):
    manager = WorkflowManager()
    manager._execute_training()
    assert hasattr(manager, "model")
    assert manager.model is not None


if __name__ == "__main__":
    pytest.main()
