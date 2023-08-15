import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from skimage import transform


def review(image, image_files, kind):
    print(f"First {kind} File Name:", image_files[0])
    print(f"{kind} Format:", image.format)
    print(f"{kind} Mode:", image.mode)
    print(f"{kind} Size:", image.size)


def preprocess_train_data(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    images = []
    labels = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file)

        image = Image.open(image_path)
        image = image.resize((256, 256))

        label = Image.open(label_path)
        label = label.resize((256, 256))

        images.append(np.array(image))
        labels.append(np.array(label))

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def check_data(images, labels):
    print("Data Range:")
    print("Images Min:", np.min(images), "Max:", np.max(images))
    print("Labels Min:", np.min(labels), "Max:", np.max(labels))

    print("\nData Types:")
    print("Images Type:", images.dtype)
    print("Labels Type:", labels.dtype)


# Function to convert image data to float32, preprocess labels, and normalize images
def preprocess_data(images, labels):
    
    images = images.astype(np.float32)
    labels = labels.astype(np.int32)
    # Convert labels to binary format (0 for background, 1 for live cells)
    binary_labels = (labels > 0).astype(np.int32)
    # Normalize the image data to the range [0, 1]
    images /= 65535

    return images, binary_labels


def data_augmentation(images, labels, num_augmented_samples, target_size=(256, 256)):
    augmented_data = []

    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]

        augmented_data.append((image, label))

        for _ in range(num_augmented_samples):
            # Apply horizontal flip
            if random.random() > 0.5:
                augmented_image = np.fliplr(image)
                augmented_label = np.fliplr(label)
                augmented_data.append((augmented_image, augmented_label))

            # Apply vertical flip
            if random.random() > 0.5:
                augmented_image = np.flipud(image)
                augmented_label = np.flipud(label)
                augmented_data.append((augmented_image, augmented_label))
            
            # Apply rotation between -90 to 90 degrees
            angle = random.uniform(-90, 90)
            augmented_image = transform.rotate(image, angle, preserve_range=True)
            augmented_label = transform.rotate(label, angle, preserve_range=True)
            augmented_data.append((augmented_image, augmented_label))

            # Apply rotation between -30 to 30 degrees
            angle = random.uniform(-30, 30)
            augmented_image = transform.rotate(image, angle, preserve_range=True)
            augmented_label = transform.rotate(label, angle, preserve_range=True)
            augmented_data.append((augmented_image, augmented_label))

            # Apply random zoom
            zoom_factor = random.uniform(0.9, 1.1)
            augmented_image = transform.rescale(image, zoom_factor, preserve_range=True)
            augmented_label = transform.rescale(label, zoom_factor, preserve_range=True)
            augmented_data.append((augmented_image, augmented_label))

            # Apply random shear
            shear_factor = random.uniform(-0.2, 0.2)
            augmented_image = transform.warp(image, transform.AffineTransform(shear=shear_factor), preserve_range=True)
            augmented_label = transform.warp(label, transform.AffineTransform(shear=shear_factor), preserve_range=True)
            augmented_data.append((augmented_image, augmented_label))

            # Apply random brightness and contrast
            augmented_image = np.clip(image + random.uniform(-0.3, 0.3), 0, 1)
            augmented_data.append((augmented_image, label))

    # Resize all augmented images and labels to the target size
    resized_augmented_data = []
    for augmented_image, augmented_label in augmented_data:
        resized_augmented_image = transform.resize(augmented_image, target_size, preserve_range=True)
        resized_augmented_label = transform.resize(augmented_label, target_size, preserve_range=True)
        resized_augmented_data.append((resized_augmented_image, resized_augmented_label))

    return resized_augmented_data