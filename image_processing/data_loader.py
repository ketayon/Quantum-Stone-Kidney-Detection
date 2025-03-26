import os
import numpy as np
import logging
from PIL import Image
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = [
    "dataset_path_stone", "dataset_path_normal",
    "X_train", "X_test", "y_train", "y_test", "count_images"
]

dataset_path_normal = "datasets/Normal"
dataset_path_stone = "datasets/stone"

# Clean .DS_Store
for folder in [dataset_path_stone, dataset_path_normal]:
    ds_store = os.path.join(folder, ".DS_Store")
    if os.path.exists(ds_store):
        os.remove(ds_store)
        log.info(f"Removed: {ds_store}")

img_size = (256, 256)


def load_images_from_folder(folder, label, target_size=img_size):
    data, labels = [], []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize(target_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            log.warning(f"Error loading image {img_name}: {e}")
    return data, labels


def load_and_limit_data(path, label, num_samples):
    data, labels = load_images_from_folder(path, label)
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    return [data[i] for i in indices], [labels[i] for i in indices]


normal_data, normal_labels = load_and_limit_data(dataset_path_normal, 0, 3500)
stone_data, stone_labels = load_and_limit_data(dataset_path_stone, 1, 5002)

all_data = np.concatenate([normal_data, stone_data])
all_labels = np.concatenate([normal_labels, stone_labels])

X_train, X_test, y_train, y_test = train_test_split(
    all_data, all_labels, test_size=0.2, random_state=42
)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

log.info(f"Loaded {len(X_train)} training and {len(X_test)} test samples.")


def count_images(directory):
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
