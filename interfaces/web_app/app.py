import os
import sys
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from monai.transforms import ScaleIntensity
from flask import Flask, render_template, request, jsonify
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.algorithms import PegasosQSVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from image_processing.data_loader import count_images
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test, dataset_path_stone, dataset_path_normal
from quantum_classification.kernel_learning import ansatz
from workflow.workflow_manager import WorkflowManager

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize workflow manager
workflow_manager = WorkflowManager()

# Load trained quantum model
MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_kidney.model"
if os.path.exists(MODEL_PATH):
    loaded_model = PegasosQSVC.load(MODEL_PATH)
    log.info("Loaded trained PegasosQSVC model.")
else:
    loaded_model = None
    log.warning("No trained model found!")


@app.route("/")
def home():
    """Render the main dashboard."""
    return render_template("index.html")


@app.route("/dataset-info")
def dataset_info():
    """Get dataset statistics."""
    stone_count = count_images(dataset_path_stone)
    normal_count = count_images(dataset_path_normal)

    return jsonify({
        "stone_count": stone_count,
        "normal_count": normal_count
    })


@app.route("/ultrasound-image")
def ultrasound_image():
    """Display an Ultrasound image and its color transformation."""

    # Check if the dataset folder exists
    if not os.path.exists(dataset_path_stone):
        log.error("Dataset folder not found!")
        return jsonify({"error": "Dataset folder not found!"}), 404

    # Get a list of available images in the stone dataset
    image_files = [f for f in os.listdir(dataset_path_stone) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        log.error("No Ultrasound images found in dataset!")
        return jsonify({"error": "No Ultrasound images found in dataset!"}), 404

    # Select a random Ultrasound image
    image_filename = random.choice(image_files)
    image_path = os.path.join(dataset_path_stone, image_filename)

    log.info(f"Using Ultrasound Image: {image_filename}")

    # Load and preprocess Ultrasound image
    ultrasound_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    transform_img = ScaleIntensity(minv=0.0, maxv=1.0)
    ultrasound_image_scaled = transform_img(ultrasound_image.astype(np.float32))

    # Normalize before applying colormap
    colored_ultrasound = plt.cm.viridis(ultrasound_image_scaled / np.max(ultrasound_image_scaled))

    # Save processed images
    gray_image_path = os.path.join(STATIC_DIR, "ultrasound_gray.jpg")
    colored_image_path = os.path.join(STATIC_DIR, "ultrasound_colored.jpg")

    plt.imsave(gray_image_path, ultrasound_image, cmap="gray")
    plt.imsave(colored_image_path, colored_ultrasound)

    return jsonify({
        "gray_image": "static/ultrasound_gray.jpg",
        "colored_image": "static/ultrasound_colored.jpg"
    })


@app.route("/pca-plot")
def pca_plot():
    """Generate and save a PCA scatter plot of the dataset."""
    plt.figure(figsize=(10, 8))

    plt.scatter(
        X_train_reduced[np.where(y_train == 0)[0], 0],
        X_train_reduced[np.where(y_train == 0)[0], 1],
        marker="s",
        facecolors="w",
        edgecolors="green",
        label="Normal Kidney (Train)",
    )

    plt.scatter(
        X_train_reduced[np.where(y_train == 1)[0], 0],
        X_train_reduced[np.where(y_train == 1)[0], 1],
        marker="o",
        facecolors="w",
        edgecolors="orange",
        label="Kidney Stone (Train)",
    )

    plt.legend()
    plt.title("PCA Dataset Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    plt.savefig(os.path.join(STATIC_DIR, "pca_plot.jpg"))
    return jsonify({"pca_plot": "static/pca_plot.jpg"})


@app.route("/predict-probabilities")
def predict_probabilities():
    """Generate and save a histogram of predicted probabilities."""
    if loaded_model is None:
        return jsonify({"error": "No trained model found"}), 404

    y_pred = loaded_model.predict(X_test_reduced[:30])
    y_pred_probabilities_tensor = torch.tensor(y_pred, dtype=torch.float32)
    y_pred_positive_probs = y_pred_probabilities_tensor.numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_positive_probs, bins=30, alpha=0.7, color="orange", label="Predicted Probabilities")
    plt.axvline(0.5, color="blue", linestyle="--", label="Decision Threshold (0.5)", linewidth=1.5)
    plt.legend()
    plt.xlabel("Predicted Probability for Kidney Stone")
    plt.ylabel("Frequency")
    plt.title("Predicted Probability Distribution")

    plt.savefig(os.path.join(STATIC_DIR, "predicted_probs.jpg"))
    return jsonify({"predicted_probs_plot": "static/predicted_probs.jpg"})


@app.route("/confusion-matrix")
def confusion_matrix_plot():
    """Generate and save a confusion matrix visualization."""
    if loaded_model is None:
        return jsonify({"error": "No trained model found"}), 404

    y_pred = loaded_model.predict(X_test_reduced[:30])
    conf_matrix = confusion_matrix(y_test[:30], y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.jpg"))
    return jsonify({"conf_matrix_plot": "static/confusion_matrix.jpg"})


@app.route("/classification-score")
def classification_score():
    """Return classification accuracy of the trained quantum model."""
    if loaded_model is None:
        return jsonify({"error": "No trained model found"}), 404

    train_score = loaded_model.score(X_train_reduced[:10], y_train[:10])
    test_score = loaded_model.score(X_test_reduced[:10], y_test[:10])

    return jsonify({
        "train_score": f"{train_score:.2f}",
        "test_score": f"{test_score:.2f}"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
