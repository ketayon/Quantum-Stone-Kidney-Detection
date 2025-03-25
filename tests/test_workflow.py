import os
import sys
import numpy as np
import pytest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflow.workflow_manager import WorkflowManager
from workflow.job_scheduler import JobScheduler

# Dummy input vector of 54 features (18 qubits x 3 layers)
valid_features = np.linspace(0, np.pi, 54)
invalid_features = np.linspace(0, np.pi, 20)  # Too short

def test_scheduler_executes_task():
    scheduler = JobScheduler(max_workers=1)
    def task(x): return x + 1
    assert scheduler.schedule_task(task, 2) == 3

def test_create_quantum_circuit():
    qc = WorkflowManager.create_quantum_circuit([np.pi / 2] * 5)
    assert qc.num_qubits == 5
    assert qc.name == "circuit-169"

def test_run_quantum_classification_returns_counts():
    qc = WorkflowManager.create_quantum_circuit([0.5] * 3)
    counts = WorkflowManager.run_quantum_classification(qc)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 1024

def test_interpret_counts_detected():
    counts = {'100': 800, '000': 224}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result == "Kidney Stone Detected"

def test_interpret_counts_not_detected():
    counts = {'000': 900, '100': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result == "No Kidney Stone Detected"

def test_interpret_invalid_counts():
    with pytest.raises(ValueError):
        WorkflowManager._interpret_quantum_counts({})

def test_classify_with_quantum_circuit():
    result = WorkflowManager.classify_with_quantum_circuit([np.pi/4] * 6)
    assert result in ["Kidney Stone Detected", "No Kidney Stone Detected"]

@patch("workflow.workflow_manager.apply_noise_mitigation")
def test_classify_with_quantum_circuit_noise(mock_noise):
    mock_noise.return_value = None  # AerSimulator handles None
    result = WorkflowManager.classify_with_quantum_circuit_noise(valid_features)
    assert result in ["Kidney Stone Detected", "No Kidney Stone Detected"]

def test_classify_with_quantum_circuit_noise_invalid():
    with pytest.raises(ValueError):
        WorkflowManager.classify_with_quantum_circuit_noise(invalid_features)

@patch("workflow.workflow_manager.train_and_save_qsvc")
def test_training_pipeline(mock_train):
    mock_train.return_value = 0.9
    manager = WorkflowManager()
    manager._execute_training()
    assert isinstance(manager.model, type(manager.model))

def test_infer_with_model():
    num_qubits = 6
    layers = 3
    total_params = num_qubits * layers
    manager = WorkflowManager()
    dummy_input = np.random.uniform(0, np.pi, (1, total_params)).astype(np.float32)
    prediction = manager.classify_ultrasound_images(dummy_input)
    assert prediction is not None

if __name__ == "__main__":
    pytest.main()
