import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from qiskit.circuit import Parameter

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflow.job_scheduler import JobScheduler
from workflow.workflow_manager import WorkflowManager

# -----------------------------
# Fixtures
# -----------------------------
valid_features = np.linspace(0, np.pi, 54)  # 18 qubits x 3 layers
invalid_features = np.linspace(0, np.pi, 10)  # Too short

# -----------------------------
# JobScheduler Tests
# -----------------------------

def test_scheduler_executes_task():
    scheduler = JobScheduler(max_workers=1)
    result = scheduler.schedule_task(lambda x: x + 1, 10)
    assert result == 11

# -----------------------------
# Interpretation Logic
# -----------------------------

def test_interpret_counts_stone():
    counts = {'111': 900, '000': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result in ["Stone", "Normal"]

def test_interpret_counts_normal():
    counts = {'000': 900, '111': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result in ["Stone", "Normal"]

def test_interpret_empty_counts():
    with pytest.raises(ValueError, match="Empty counts from quantum simulation."):
        WorkflowManager._interpret_quantum_counts({})

# -----------------------------
# Local Classification
# -----------------------------

@patch("workflow.workflow_manager.predict_with_expectation", return_value="Stone")
def test_classify_with_quantum_circuit(mock_predict):
    result = WorkflowManager.classify_with_quantum_circuit(valid_features)
    assert result == "Stone"

def test_classify_with_quantum_circuit_noise_invalid():
    with pytest.raises(ValueError, match="Expected .* features"):
        WorkflowManager.classify_with_quantum_circuit_noise(invalid_features)

# -----------------------------
# Initialization / Load Logic
# -----------------------------

@patch("workflow.workflow_manager.PegasosQSVC.load")
@patch("workflow.workflow_manager.os.path.exists", return_value=True)
def test_workflow_manager_loads_existing_model(mock_exists, mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    manager = WorkflowManager()
    assert manager.model == mock_model
    mock_load.assert_called_once()

@patch("workflow.workflow_manager.train_and_save_qsvc", return_value=0.95)
@patch("workflow.workflow_manager.os.path.exists", return_value=False)
def test_workflow_manager_trains_when_model_missing(mock_exists, mock_train):
    with patch("workflow.workflow_manager.PegasosQSVC.save") as mock_save:
        manager = WorkflowManager()
        assert manager.model is not None


if __name__ == "__main__":
    pytest.main()
