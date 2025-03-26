import os
import logging
from collections import Counter
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import PegasosQSVC

from quantum_classification.quantum_model import pegasos_svc, train_and_save_qsvc
from quantum_classification.noise_mitigation import apply_noise_mitigation
from workflow.job_scheduler import JobScheduler
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params
from quantum_classification.quantum_estimation import predict_with_expectation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "PegasosQSVC_Fidelity_quantm_trainer_kidney.model")

class WorkflowManager:
    """Manages the Kidney Stone Quantum Classification Workflow"""

    def __init__(self):
        self.job_scheduler = JobScheduler()
        self.model = None

        log.info("🧬 Quantum Kidney Stone Workflow Initialized (Local Simulator)")
        self._ensure_model_dir()
        self._load_or_train_model()

    def _ensure_model_dir(self):
        """Ensure the models directory exists."""
        if not os.path.exists(MODEL_DIR):
            log.warning("Model directory not found. Creating: %s", MODEL_DIR)
            os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.listdir(MODEL_DIR):
            log.warning("Model directory is empty. If running Docker, mount volume: -v $(pwd)/models:/app/models")

    def _load_or_train_model(self):
        """Load trained model or train if not available."""
        if os.path.exists(MODEL_PATH):
            log.info("📦 Loading pre-trained Quantum Kidney Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("✅ Model loaded.")
        else:
            log.info("⚠️ No pre-trained model found. Training a new one...")
            self.train_quantum_model()
            log.info("💾 Saving trained model to: %s", MODEL_PATH)
            pegasos_svc.save(MODEL_PATH)
            self.model = pegasos_svc
            log.info("✅ Model saved.")

    def train_quantum_model(self):
        log.info("🧠 Scheduling Quantum Model Training...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        log.info("🚀 Executing Quantum Kidney Model Training...")
        accuracy = train_and_save_qsvc()
        self.model = pegasos_svc
        log.info(f"🎯 Training Complete. Accuracy: {accuracy}")

    def classify_kidney_mri(self, image_data):
        if self.model is None:
            log.error("❌ No trained model found. Train the model first.")
            return None
        log.info("📊 Scheduling QSVC-based classification...")
        return self.job_scheduler.schedule_task(self._infer_kidney_stone, image_data)

    def _infer_kidney_stone(self, image_data):
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
        num_qubits, layers = 18, 3
        total_params = num_qubits * layers

        if len(image_features) != total_params:
            raise ValueError(f"Expected {total_params} features, got {len(image_features)}")

        params = [Parameter(f"θ{i}") for i in range(total_params)]
        ansatz = build_ansatz(num_qubits, params)
        qc = ansatz.assign_parameters(dict(zip(params, image_features)))
        qc.measure_all()

        simulator = AerSimulator()
        noise_model = apply_noise_mitigation(simulator)
        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=1024).result()
        counts = result.get_counts()

        return WorkflowManager._interpret_quantum_counts(counts)

    @staticmethod
    def classify_with_quantum_circuit(image_features):
        log.info("🧪 Running expectation-based classification (local)...")
        prediction = predict_with_expectation(image_features)
        log.info(f"🔬 Quantum Estimation Prediction: {prediction}")
        return prediction
