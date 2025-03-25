import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from werkzeug.utils import secure_filename
import tempfile
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import cv2
import logging
from monai.transforms import ScaleIntensity
from flask import Flask, render_template, request, jsonify
from qiskit_machine_learning.algorithms import PegasosQSVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from image_processing.data_loader import count_images
from image_processing.image_transformations import apply_grayscale, apply_gaussian_blur
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced, MinMaxScaler, reduce_to_n_dimensions
from image_processing.data_loader import y_train, y_test, dataset_path_stone, dataset_path_normal
from workflow.workflow_manager import WorkflowManager


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

workflow_manager = WorkflowManager()

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

    if not os.path.exists(dataset_path_stone):
        log.error("Dataset folder not found!")
        return jsonify({"error": "Dataset folder not found!"}), 404

    image_files = [f for f in os.listdir(dataset_path_stone) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        log.error("No Ultrasound images found in dataset!")
        return jsonify({"error": "No Ultrasound images found in dataset!"}), 404

    image_filename = random.choice(image_files)
    image_path = os.path.join(dataset_path_stone, image_filename)
    log.info(f"Using Ultrasound Image: {image_filename}")

    ultrasound_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    transform_img = ScaleIntensity(minv=0.0, maxv=1.0)
    ultrasound_image_scaled = transform_img(ultrasound_image.astype(np.float32))
    colored_ultrasound = plt.cm.viridis(ultrasound_image_scaled / np.max(ultrasound_image_scaled))

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

    plot_path = os.path.join(STATIC_DIR, "pca_plot.jpg")
    plt.savefig(plot_path)
    return jsonify({"pca_plot": "static/pca_plot.jpg"})


@app.route("/predict-probabilities")
def predict_probabilities():
    """Generate and save a histogram of predicted probabilities."""
    try:
        y_pred = workflow_manager.model.predict(X_test_reduced[:30])
        y_pred_probabilities_tensor = torch.tensor(y_pred, dtype=torch.float32)
        y_pred_positive_probs = y_pred_probabilities_tensor.numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_positive_probs, bins=30, alpha=0.7, color="orange", label="Predicted Probabilities")
        plt.axvline(0.5, color="blue", linestyle="--", label="Decision Threshold (0.5)", linewidth=1.5)
        plt.legend()
        plt.xlabel("Predicted Probability for Kidney Stone")
        plt.ylabel("Frequency")
        plt.title("Predicted Probability Distribution")

        plot_path = os.path.join(STATIC_DIR, "predicted_probs.jpg")
        plt.savefig(plot_path)
        return jsonify({"predicted_probs_plot": "static/predicted_probs.jpg"})

    except Exception as e:
        log.exception("Error generating prediction probabilities.")
        return jsonify({"error": str(e)}), 500


@app.route("/confusion-matrix")
def confusion_matrix_plot():
    """Generate and save a confusion matrix visualization."""
    try:
        y_pred = workflow_manager.model.predict(X_test_reduced[:30])
        conf_matrix = confusion_matrix(y_test[:30], y_pred)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        plot_path = os.path.join(STATIC_DIR, "confusion_matrix.jpg")
        plt.savefig(plot_path)
        return jsonify({"conf_matrix_plot": "static/confusion_matrix.jpg"})

    except Exception as e:
        log.exception("Error generating confusion matrix.")
        return jsonify({"error": str(e)}), 500


@app.route("/classification-score")
def classification_score():
    """Return classification accuracy of the trained quantum model."""
    try:
        train_score = workflow_manager.model.score(X_train_reduced[:10], y_train[:10])
        test_score = workflow_manager.model.score(X_test_reduced[:10], y_test[:10])

        return jsonify({
            "train_score": f"{train_score:.2f}",
            "test_score": f"{test_score:.2f}"
        })
    except Exception as e:
        log.exception("Error computing classification score.")
        return jsonify({"error": str(e)}), 500


@app.route("/quantum-circuit-classify", methods=["POST"])
def quantum_circuit_classify():
    """
    Classify an ultrasound image using quantum circuit classification.
    Expects JSON: { "features": [feature_vector] }
    """
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = np.array(data["features"], dtype=np.float32)
        prediction = workflow_manager.classify_with_quantum_circuit(features)

        return jsonify({
            "quantum_circuit_prediction": prediction
        })

    except Exception as e:
        log.exception("Quantum Circuit Classification Error")
        return jsonify({"error": str(e)}), 500


@app.route("/quantum-circuit-classify-noise", methods=["POST"])
def quantum_circuit_classify_noise():
    """
    Classify an ultrasound image using quantum circuit with realistic noise.
    Expects JSON: { "features": [feature_vector] }
    """
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = np.array(data["features"], dtype=np.float32)
        prediction = workflow_manager.classify_with_quantum_circuit_noise(features)

        return jsonify({
            "quantum_circuit_prediction_with_noise": prediction
        })

    except Exception as e:
        log.exception("Noise-aware Quantum Classification Error")
        return jsonify({"error": str(e)}), 500


@app.route("/classify-image", methods=["POST"])
def classify_uploaded_image():
    """
    Classify an uploaded image using a quantum circuit (noise-aware).
    Returns: JSON with quantum_prediction or error.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected image"}), 400

    try:
        filename = secure_filename(image.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        image.save(temp_path)

        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("Invalid or unreadable image.")

        gray = apply_grayscale(img)
        blurred = apply_gaussian_blur(gray)
        resized = cv2.resize(blurred, (256, 256)).flatten().astype(np.float32)

        if np.std(resized) < 1e-3:
            raise ValueError("Image content too uniform — likely invalid or blank.")

        num_qubits = 18 
        layers = 3
        total_params = num_qubits * layers

        reduced = reduce_to_n_dimensions(resized.reshape(1, -1), num_qubits)

        scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(reduced).flatten()

        features = np.tile(scaled, layers)[:total_params]

        if len(features) != total_params:
            raise ValueError(f"Expected {total_params} features, got {len(features)}.")

        prediction = workflow_manager.classify_with_quantum_circuit_noise(features)

        return jsonify({"quantum_prediction": prediction})

    except Exception as e:
        log.exception("Image classification error")
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
