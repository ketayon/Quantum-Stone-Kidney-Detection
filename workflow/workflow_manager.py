import os
import logging
from collections import Counter
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC

from quantum_classification.quantum_model import pegasos_svc, train_and_save_qsvc
from quantum_classification.noise_mitigation import apply_noise_mitigation
from workflow.job_scheduler import JobScheduler
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params
from quantum_classification.quantum_async_jobs import (
    submit_quantum_job,
    check_quantum_job
)
from quantum_classification.quantum_estimation import predict_with_expectation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_kidney.model"

token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("ERROR: QISKIT_IBM_TOKEN environment variable is not set!")

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)

backend = service.least_busy(operational=True, simulator=False)

class WorkflowManager:
    """Manages the Kidney Stone Quantum Classification Workflow"""

    def __init__(self):
        self.job_scheduler = JobScheduler()
        self.model = None
        log.info("Quantum Kidney Stone Workflow Initialized on Backend: %s", backend)
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load the trained model if it exists, otherwise train and save"""
        if os.path.exists(MODEL_PATH):
            log.info("📦 Loading pre-trained Quantum Kidney Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("✅ Model loaded successfully.")
        else:
            log.info("⚠️ No pre-trained model found. Training a new model...")
            self.train_quantum_model()
            log.info("💾 Saving trained model...")
            pegasos_svc.save(MODEL_PATH)
            self.model = pegasos_svc
            log.info("✅ Model saved at: %s", MODEL_PATH)

    def train_quantum_model(self):
        """Train the Quantum Model using Job Scheduler"""
        log.info("⏳ Scheduling Quantum Model Training...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        """Handles Quantum Training Execution"""
        log.info("🚀 Executing Quantum Kidney Stone Model Training...")
        accuracy = train_and_save_qsvc()
        self.model = pegasos_svc
        log.info(f"✅ Training Complete. Accuracy: {accuracy}")

    def classify_kidney_mri(self, image_data):
        """Classify Kidney MRI Ultrasound Images using the trained model"""
        if self.model is None:
            log.error("❌ No trained model found. Please train the model first.")
            return None
        log.info("📊 Scheduling QSVC-based classification...")
        return self.job_scheduler.schedule_task(self._infer_kidney_stone, image_data)

    def _infer_kidney_stone(self, image_data):
        """Infer using PegasosQSVC"""
        log.info("🔍 Performing QSVC Classification...")
        prediction = self.model.predict(image_data)
        return prediction
    
    @staticmethod
    def create_quantum_circuit(features):
        num_qubits = len(features)
        params = [Parameter(f"θ{i}") for i in range(num_qubits)]
        circuit = build_ansatz(num_qubits, params)
        param_dict = dict(zip(params, features))
        qc = circuit.assign_parameters(param_dict)
        qc.measure_all()
        return qc

    @staticmethod
    def run_quantum_classification(qc):
        simulator = AerSimulator()
        transpiled_qc = transpile(qc, simulator)
        result = simulator.run(transpiled_qc, shots=1024).result()
        return result.get_counts()

    @staticmethod
    def _interpret_quantum_counts(counts):
        if not counts:
            raise ValueError("Empty counts from quantum simulation.")
        most_common = Counter(counts).most_common(1)[0][0]
        bit = int(most_common[::-1][0])
        return "Stone" if bit else "Normal"

    @staticmethod
    def classify_with_quantum_circuit_noise(image_features):
        num_qubits = 18
        layers = 3
        total_params = num_qubits * layers

        if len(image_features) != total_params:
            raise ValueError(f"Expected {total_params} features, got {len(image_features)}")

        params = [Parameter(f"θ{i}") for i in range(total_params)]
        ansatz = build_ansatz(num_qubits, params)
        param_dict = dict(zip(params, image_features))
        qc = ansatz.assign_parameters(param_dict)
        qc.measure_all()

        noise_model = apply_noise_mitigation(backend)
        simulator = AerSimulator(noise_model=noise_model)
        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=1024).result()
        counts = result.get_counts()

        return WorkflowManager._interpret_quantum_counts(counts)

    @staticmethod
    def classify_with_quantum_circuit(image_features):
        """
        Classify image using quantum circuit expectation value (blocking).
        Used for CLI or internal testing, not recommended for web apps.
        """
        log.info("🧪 Running synchronous expectation-based classification...")
        prediction = predict_with_expectation(image_features)
        log.info(f"🔬 Quantum Estimation Prediction: {prediction}")
        return prediction

    @staticmethod
    def submit_quantum_job_async(image_features):
        """
        Submit a quantum job to IBM Quantum backend (non-blocking).
        Used for async web inference.
        Returns: job_id (str)
        """
        log.info("📡 Submitting async quantum job to IBM Quantum...")
        return submit_quantum_job(image_features)

    @staticmethod
    def check_quantum_job_result(job_id):
        """
        Poll quantum job result.
        Returns: dict with keys: status, prediction, expectation_value (if complete)
        """
        log.info(f"🔁 Checking status of job: {job_id}")
        return check_quantum_job(job_id)
