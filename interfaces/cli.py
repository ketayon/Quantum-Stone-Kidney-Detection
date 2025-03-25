import argparse
import logging
import os
import sys
import numpy as np
from qiskit_machine_learning.algorithms import PegasosQSVC

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(BASE_DIR)

from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test, dataset_path_normal, dataset_path_stone
from quantum_classification.quantum_model import pegasos_svc
from workflow.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

workflow_manager = WorkflowManager()


def count_images(directory):
    """Counts number of image files in the directory."""
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])


def view_dataset_info():
    """Display number of kidney stone vs normal ultrasound images in dataset."""
    tumor_count = count_images(dataset_path_stone)
    normal_count = count_images(dataset_path_normal)
    log.info("🧬 Dataset Information:")
    log.info(f"   Kidney Stone Images : {tumor_count}")
    log.info(f"   Normal Kidney Images: {normal_count}")


def show_model_scores():
    """Evaluate model accuracy on train and test sets."""
    try:
        # Check if model is trained (fit) by attempting a small prediction
        _ = pegasos_svc.predict(X_train_reduced[:1])
    except Exception:
        log.info("🔄 PegasosQSVC model not trained yet. Training now...")
        from quantum_classification.quantum_model import train_and_save_qsvc
        train_and_save_qsvc()

    train_score = pegasos_svc.score(X_train_reduced, y_train)
    test_score = pegasos_svc.score(X_test_reduced, y_test)
    log.info(f"📊 Quantum QSVC Train Accuracy: {train_score:.2f}")
    log.info(f"📊 Quantum QSVC Test Accuracy : {test_score:.2f}")


def predict_sample():
    """Allow user to classify a new PCA-reduced ultrasound feature vector."""
    log.info("📌 Enter 18 PCA-reduced features (comma-separated):")
    raw_input = input("> ").strip()

    try:
        features = np.array([float(x) for x in raw_input.split(",")], dtype=np.float32)
        if len(features) != 18:
            raise ValueError("Expected 18 features (matching 18 qubits).")

        prediction = workflow_manager.classify_with_quantum_circuit_noise(
            np.tile(features, 3)  # Expand to match 3 layers = 54 total features
        )
        log.info(f"Quantum Prediction: {prediction}")
    except Exception as e:
        log.error(f"Error during classification: {e}")


def main():
    parser = argparse.ArgumentParser(description="🧠 CLI for Quantum Kidney Stone Detection")

    parser.add_argument("--dataset-info", action="store_true", help="Show ultrasound dataset statistics")
    parser.add_argument("--model-score", action="store_true", help="Display quantum model performance")
    parser.add_argument("--predict", action="store_true", help="Predict using a PCA-reduced feature vector")

    args = parser.parse_args()

    if args.dataset_info:
        view_dataset_info()
    elif args.model_score:
        show_model_scores()
    elif args.predict:
        predict_sample()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
