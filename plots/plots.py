import numpy as np
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def plotAll(first_image, first_label, first_border, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(9, 4))

    axs[0].imshow(first_image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(first_label)
    axs[1].set_title("Segmentation Label")
    axs[1].axis("off")

    axs[2].imshow(first_border)
    axs[2].set_title("Border")
    axs[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "plot_first_image_label_border.png"), dpi=120, bbox_inches='tight')
    
    plt.show()

def plotImageLabel(images, labels, num_to_display, plot_name="plot", save_path=None):
    fig, axs = plt.subplots(2, num_to_display, figsize=(9, 6))
    for i in range(num_to_display):
        axs[0, i].imshow(images[i])
        axs[0, i].set_title(f"Image {i+1}")
        axs[0, i].axis("off")

        axs[1, i].imshow(labels[i], cmap='gray')
        axs[1, i].set_title(f"Label {i+1}")
        axs[1, i].axis("off")

    plt.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{plot_name}_images_labels.png"), dpi=120, bbox_inches='tight')
    plt.show()


# Define a function to display a grid of images
def plot_image_grid(images, titles, rows, cols, plot_name, save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{plot_name}_augmented_images.png"), dpi=100, bbox_inches='tight')
    plt.show()
    
    
def plot_metrics(train_losses, train_ious, train_accuracies, val_losses, val_ious, val_accuracies, save_path=None):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_ious) + 1), train_ious, label='Train IoU')
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"Model_metrics.png"), dpi=100, bbox_inches='tight')
    plt.show()


def visualize_random_test_images(test_folder, test_files, num_images=3, save_path=None):
    random_numbers = [random.randint(0, len(test_files) - 1) for _ in range(num_images)]
    
    fig, axs = plt.subplots(1, num_images, figsize=(10, 4))
    
    for i, random_number in enumerate(random_numbers):
        image_path = os.path.join(test_folder, test_files[random_number])
        test_image = Image.open(image_path)
        axs[i].imshow(test_image)
        axs[i].set_title(f"Random Test Image {i+1}")
        axs[i].axis("off")
    
    plt.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"random_test_images.png"), dpi=100, bbox_inches='tight')
    plt.show()
    
def visualize_predicted_masks(test_images, predicted_masks_array, num_samples=10, save_path=None):
    for _ in range(num_samples):
        idx = random.randint(0, len(test_images) - 1)
        test_image = test_images[idx]
        predicted_mask = predicted_masks_array[idx]

        # Invert the predicted mask (background becomes black, masks become white)
        inverted_predicted_mask = 1 - predicted_mask

        # Plot the test image and inverted predicted mask
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.title(f'Test Image {idx}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(inverted_predicted_mask, cmap='binary', vmin=0, vmax=1)
        plt.title(f'Predicted Mask {idx}')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f"random_predicted_images_masks.png"), dpi=140, bbox_inches='tight')
        plt.show()

