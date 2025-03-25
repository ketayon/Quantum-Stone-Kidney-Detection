import os
import logging
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params

log = logging.getLogger(__name__)

# Ensure IBM Quantum token is available
token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("QISKIT_IBM_TOKEN environment variable is not set!")

# Setup IBM Runtime and backend
service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)
backend = service.least_busy(operational=True, simulator=False)
estimator = Estimator(mode=backend)

# Temporary in-memory job store (use DB/Redis in production)
job_store = {}

def submit_quantum_job(features):
    """
    Submit an async IBM Quantum job to classify kidney stone from ultrasound features.

    Args:
        features (List[float]): Flattened and reduced quantum features (length: num_qubits * layers).

    Returns:
        str: Job ID to use for polling status/results.
    """
    num_qubits = 18
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)

    if len(features) != total_params:
        raise ValueError(f"Expected {total_params} features, got {len(features)}")

    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)

    observable = SparsePauliOp("Z" * num_qubits)

    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pass_manager.run(circuit)
    observable = observable.apply_layout(transpiled.layout)

    job = estimator.run([(transpiled, observable, [features])])
    job_id = job.job_id()
    job_store[job_id] = job

    log.info(f"📡 Submitted IBM Quantum job for Kidney Stone Detection: {job_id}")
    return job_id


def check_quantum_job(job_id, threshold=0.5):
    """
    Check the status or result of a previously submitted IBM Quantum job.

    Args:
        job_id (str): Job identifier returned from `submit_quantum_job`.
        threshold (float): Decision threshold for expectation value.

    Returns:
        dict: Status and result (pending/complete/error).
    """
    job = job_store.get(job_id)
    if job is None:
        return {"status": "error", "message": f"Job ID {job_id} not found"}

    if not job.done():
        return {"status": "pending", "message": "Job is still running..."}

    result = job.result()
    value = float(result[0].data.evs)
    prediction = "Kidney Stone Detected" if value > threshold else "No Kidney Stone"

    return {
        "status": "complete",
        "expectation_value": round(value, 4),
        "prediction": prediction
    }
