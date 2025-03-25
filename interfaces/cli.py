import argparse
import logging
import os
import sys
import numpy as np
from qiskit_machine_learning.algorithms import PegasosQSVC

# Setup project path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))

from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test, dataset_path_stone, dataset_path_normal
from workflow.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize the workflow manager (handles both QSVC and circuit models)
workflow_manager = WorkflowManager()


def count_images(directory):
    """Counts number of images in a directory."""
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])


def view_dataset_info():
    """Displays dataset statistics (Kidney Stone vs Normal)."""
    stone_count = count_images(dataset_path_stone)
    normal_count = count_images(dataset_path_normal)
    log.info("🩺 Dataset Information:")
    log.info(f"  - Kidney Stone Images: {stone_count}")
    log.info(f"  - Normal Kidney Images: {normal_count}")


def show_model_scores():
    """Displays Quantum Model accuracy on Train & Test sets."""
    train_score = workflow_manager.model.score(X_train_reduced, y_train)
    test_score = workflow_manager.model.score(X_test_reduced, y_test)

    log.info("📊 PegasosQSVC Model Scores:")
    log.info(f"  - Train Accuracy: {train_score:.2f}")
    log.info(f"  - Test Accuracy : {test_score:.2f}")


def predict_with_qsvc():
    """Predicts kidney stone using QSVC model for a sample input."""
    log.info("Enter reduced image feature vector (comma-separated):")
    try:
        user_input = input("> ").strip()
        features = np.array([float(x) for x in user_input.split(",")]).reshape(1, -1)
        prediction = workflow_manager.classify_ultrasound_images(features)
        log.info(f"QSVC Prediction: {prediction}")
    except Exception as e:
        log.error(f"❌ Error during QSVC classification: {e}")


def predict_with_quantum_circuit():
    """Predicts kidney stone using quantum circuit-based classification."""
    log.info("Enter reduced image feature vector (comma-separated):")
    try:
        user_input = input("> ").strip()
        features = np.array([float(x) for x in user_input.split(",")])
        prediction = workflow_manager.classify_with_quantum_circuit(features)
        log.info(f"Quantum Circuit Prediction: {prediction}")
    except Exception as e:
        log.error(f"❌ Error during quantum circuit classification: {e}")


def main():
    parser = argparse.ArgumentParser(description="🧪 Quantum CLI for Kidney Stone Detection")
    parser.add_argument("--dataset-info", action="store_true", help="Show dataset statistics")
    parser.add_argument("--model-score", action="store_true", help="Show PegasosQSVC model accuracy")
    parser.add_argument("--predict-qsvc", action="store_true", help="Classify with PegasosQSVC model")
    parser.add_argument("--predict-circuit", action="store_true", help="Classify with quantum circuit-based classifier")

    args = parser.parse_args()

    if args.dataset_info:
        view_dataset_info()
    elif args.model_score:
        show_model_scores()
    elif args.predict_qsvc:
        predict_with_qsvc()
    elif args.predict_circuit:
        predict_with_quantum_circuit()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
