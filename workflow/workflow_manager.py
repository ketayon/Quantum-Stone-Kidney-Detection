import os
import logging
from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC

from quantum_classification.quantum_model import pegasos_svc, train_and_save_qsvc
from workflow.job_scheduler import JobScheduler
from quantum_classification.noise_mitigation import apply_noise_mitigation
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params


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
            log.info("Loading pre-trained Quantum Kidney Stone Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("Model loaded successfully.")
        else:
            log.info("No pre-trained model found. Training a new model...")
            self.train_quantum_model()
            log.info("Saving trained model...")
            pegasos_svc.save(MODEL_PATH)
            self.model = pegasos_svc
            log.info("Model saved at: %s", MODEL_PATH)

    def train_quantum_model(self):
        """Train the Quantum Model using Job Scheduler"""
        log.info("Scheduling Quantum Model Training...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        """Handles Quantum Training Execution"""
        log.info("Executing Quantum Kidney Stone Model Training...")
        accuracy = train_and_save_qsvc()
        self.model = pegasos_svc
        log.info(f"Quantum Kidney Stone Model Training Completed. Accuracy: {accuracy}")

    def classify_ultrasound_images(self, image_data):
        """Classify Ultrasound Images using the trained model"""
        if self.model is None:
            log.error("No trained model found. Please train the model first.")
            return None
        log.info("Scheduling QSVC-based classification...")
        return self.job_scheduler.schedule_task(self._infer_kidney_stone, image_data)

    def _infer_kidney_stone(self, image_data):
        """Infer using PegasosQSVC"""
        log.info("Performing QSVC Classification...")
        prediction = self.model.predict(image_data)
        return prediction

    def classify_with_quantum_circuit(self, image_features):
        """
        Classify ultrasound image using quantum circuit simulation.
        Returns: "Kidney Stone Detected" or "No Kidney Stone Detected"
        """
        log.info("Creating quantum circuit from image features...")
        qc = self.create_quantum_circuit(image_features)

        log.info("Running quantum circuit simulation...")
        counts = self.run_quantum_classification(qc)

        prediction = self._interpret_quantum_counts(counts)
        log.info("Quantum Circuit Prediction: %s", prediction)
        return prediction

    @staticmethod
    def create_quantum_circuit(image_features):
        """Create quantum circuit encoding image features using ry rotations."""
        num_qubits = len(image_features)
        qc = QuantumCircuit(num_qubits)

        for i, feature in enumerate(image_features):
            qc.h(i)
            qc.ry(float(feature), i)

        qc.measure_all()
        return qc
    
    @staticmethod
    def run_quantum_classification(qc):
        """Simulate a clean quantum circuit (no noise)."""
        log.info("Running clean quantum circuit simulation (no noise)...")
        simulator = AerSimulator()
        transpiled_qc = transpile(qc, simulator)
        result = simulator.run(transpiled_qc, shots=1024).result()
        counts = result.get_counts()
        log.info(f"Clean simulation complete. Counts: {counts}")
        return counts  # ✅ Return counts, not interpretation


    @staticmethod
    def classify_with_quantum_circuit(image_features):
        """
        Classify ultrasound image using quantum circuit simulation.
        Returns: "Kidney Stone Detected" or "No Kidney Stone Detected"
        """
        log.info("Creating quantum circuit from image features...")
        qc = WorkflowManager.create_quantum_circuit(image_features)

        log.info("Running quantum circuit simulation...")
        counts = WorkflowManager.run_quantum_classification(qc)

        prediction = WorkflowManager._interpret_quantum_counts(counts)
        log.info("Quantum Circuit Prediction: %s", prediction)
        return prediction


    @staticmethod
    def classify_with_quantum_circuit_noise(image_features):
        """Classify using custom ansatz & simulate with realistic hardware noise."""
        log.info("Creating quantum circuit with custom ansatz...")

        num_qubits = 18
        layers = 3
        total_params = num_qubits * layers

        if len(image_features) != total_params:
            raise ValueError(f"Expected {total_params} features for {num_qubits} qubits, got {len(image_features)}.")

        params = [Parameter(f"θ{i}") for i in range(total_params)]
        ansatz = build_ansatz(num_qubits, params)

        param_dict = dict(zip(params, image_features))
        qc = ansatz.assign_parameters(param_dict)
        qc.measure_all()

        noise_model = apply_noise_mitigation(backend)
        log.info(f"Simulating with noise model from backend: {backend.name}")

        simulator = AerSimulator(noise_model=noise_model)
        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=1024).result()
        counts = result.get_counts()

        log.info(f"Noise-aware simulation counts: {counts}")
        return WorkflowManager._interpret_quantum_counts(counts)


    @staticmethod
    def _interpret_quantum_counts(counts):
        """Interpret quantum circuit results by majority vote on first qubit."""
        if not counts:
            raise ValueError("Empty counts from quantum simulation.")

        most_common = Counter(counts).most_common(1)[0][0]
        log.info(f"Most common measurement result: '{most_common}'")

        # Find first valid bit (0 or 1)
        for char in most_common:
            if char in ('0', '1'):
                majority_bit = int(char)
                return "Kidney Stone Detected" if majority_bit else "No Kidney Stone Detected"

        log.error(f"Invalid bitstring from quantum circuit: '{most_common}'")
        log.error(f"All counts: {counts}")
        raise ValueError(f"Could not interpret quantum measurement result: {most_common}")
