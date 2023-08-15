import os
import numpy as np
from PIL import Image


def preprocess_test_data(test_folder):    
    
    test_files = os.listdir(test_folder)
    test_images = []
    
    for image_file in test_files:
        image_path = os.path.join(test_folder, image_file)

        # Load and preprocess the image. Resize to a common size (256*256) for U-Net
        image = Image.open(image_path)
        image = image.resize((256, 256))
        test_images.append(np.array(image))

    test_images = np.array(test_images)
    return test_images


def check_test_data(images):
    print("Data Range:")
    print("Images Min:", np.min(images), "Max:", np.max(images))

    print("\nData Types:")
    print("Images Type:", images.dtype)

    
def normalize_test_data(images):
    images = images.astype(np.float32)
    # Normalize the image data to the range [0, 1]
    images /= 65535
    return images