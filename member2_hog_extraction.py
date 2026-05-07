import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from tqdm import tqdm

DATADIR = "PetImages"

CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 128

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (32, 32),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "visualize": False
}

def extract_hog_features(image, config=HOG_CONFIG):

    features = hog(
        image,
        orientations=config["orientations"],
        pixels_per_cell=config["pixels_per_cell"],
        cells_per_block=config["cells_per_block"],
        block_norm=config["block_norm"],
        visualize=config["visualize"],
        feature_vector=True
    )

    return features

def extract_hog_batch(images, config=HOG_CONFIG):

    hog_features = []

    for image in tqdm(images, desc="Extracting HOG Features"):

        features = extract_hog_features(image, config)

        hog_features.append(features)

    return hog_features

def visualize_hog(image, save_path=None):

    features, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title("HOG Visualization")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def load_dataset(max_images_per_category=None):

    images = []
    labels = []

    for label, category in enumerate(CATEGORIES):

        path = os.path.join(DATADIR, category)

        image_files = os.listdir(path)

        if max_images_per_category:
            image_files = image_files[:max_images_per_category]

        for img_name in image_files:

            img_path = os.path.join(path, img_name)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Could not load: {img_path}")
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            images.append(image)

            labels.append(label)

    return images, labels

def validate_hog_dimensions(features):

    expected_length = 324

    if len(features) == expected_length:
        print("HOG dimension validation PASSED")
    else:
        print("HOG dimension validation FAILED")
        print("Expected:", expected_length)
        print("Got:", len(features))

if __name__ == "__main__":

    all_images, all_labels = load_dataset(
        max_images_per_category=5
    )

    print("Total Images Loaded:", len(all_images))

    print("Labels:", all_labels)

    hog_vectors = extract_hog_batch(all_images)

    validate_hog_dimensions(hog_vectors[0])

    print("Feature Vector Length:", len(hog_vectors[0]))

    visualize_hog(
        all_images[0],
        save_path="output/hog_visualizations/cat_hog.png"
    )