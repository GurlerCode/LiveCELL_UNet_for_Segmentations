import imageio
import os
import torch
import numpy as np

def predict_masks(model, test_images_tensor, threshold=0.5, device="cuda", save_folder=None):
    predicted_masks = []

    # Process each test image
    for i, test_image in enumerate(test_images_tensor):
        # Add a batch dimension to the test image tensor
        test_image_batch = test_image.unsqueeze(0).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            model_output = model(test_image_batch)
            predicted_mask = torch.sigmoid(model_output) > threshold
        
        # Convert predicted mask to numpy array
        predicted_mask = predicted_mask.cpu().squeeze().numpy()
        predicted_masks.append(predicted_mask)

        # Save predicted mask as a TIFF file if save_folder is provided
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"predicted_mask_{i}.tif")
            imageio.imwrite(save_path, predicted_mask.astype(np.uint8) * 255)

    # Convert the list of predicted masks to a numpy array
    predicted_masks_array = np.array(predicted_masks)
    return predicted_masks_array

