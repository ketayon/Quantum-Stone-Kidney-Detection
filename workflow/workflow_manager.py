import os
import logging
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC
from quantum_classification.quantum_model import pegasos_svc
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test
from workflow.job_scheduler import JobScheduler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define model path
MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_kidney.model"

# Load IBM Quantum Service
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
        """Initialize Workflow Manager"""
        self.job_scheduler = JobScheduler()
        self.model = None
        log.info("Quantum Kidney Stone Workflow Initialized on Backend: %s", backend)
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load the trained model if it exists, otherwise train and save"""
        if os.path.exists(MODEL_PATH):
            log.info("Loading pre-trained Quantum Kidney Stone Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("Model loaded successfully!")
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
        pegasos_svc.fit(X_train_reduced, y_train)
        accuracy = pegasos_svc.score(X_test_reduced, y_test)
        log.info(f"Quantum Kidney Stone Model Training Completed. Accuracy: {accuracy}")

    def classify_ultrasound_images(self, image_data):
        """Classify Ultrasound Images using the trained model"""
        if self.model is None:
            log.error("No trained model found. Please train the model first.")
            return None
        log.info("Scheduling Kidney Stone Classification Task...")
        return self.job_scheduler.schedule_task(self._infer_kidney_stone, image_data)

    def _infer_kidney_stone(self, image_data):
        """Infer if an ultrasound image contains a kidney stone"""
        log.info("Performing Kidney Stone Image Classification...")
        prediction = self.model.predict(image_data)
        return prediction
