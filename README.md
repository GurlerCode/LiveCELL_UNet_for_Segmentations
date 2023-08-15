# UNet Segmentation for LiveCELL Imaging
## IS THIS CELL ALIVE? U-Net Segmentation for LiveCell Imaging
This is the U-Net Segmentation for LiveCell Imaging project! This repository contains a comprehensive implementation of the U-Net architecture for semantic segmentation of live cell microscopy images. The U-Net model is widely used for image segmentation tasks and particularly effectively segment biomedical images with complex structures.

## Introduction
This project focuses on utilizing the U-Net architecture to perform semantic segmentation on microscopy images, specifically targeting live cell imaging data. The goal is to accurately segment different cell structures within these images, allowing for precise analysis and understanding of cellular behavior. The pipeline includes data preprocessing, augmentation, model training, testing, and evaluation.
This study was conducted as a final exam for the course Python for Engineering Data Analysis at the Technical University of Munich.

## Key Features
- **Data Preparation:** Load, preprocess, and augment training and validation data. Normalize and reshape the images and labels for input to the U-Net model.
- **U-Net Model Architecture:** Implement the U-Net architecture with encoder and decoder components. Configure loss functions, optimizers, and learning rate schedulers.
- **Training and Validation:** Train the U-Net model using mini-batch gradient descent. Monitor training progress and evaluate the model on validation data to prevent overfitting.
- **Testing and Evaluation:** Load the trained model, preprocess test data, and perform segmentation on test images. Evaluate segmentation performance using various metrics.
- **Visualization:** Visualize input images, ground truth labels, predicted masks, and intermediate results using plotting functions.
- **Results and Discussion:** Analyze and discuss the performance of the U-Net model, highlighting its accuracy and potential applications.

## LiveCell Imaging Dataset
The LiveCell Imaging dataset is a collection of microscopy images designed for the purpose of cell segmentation and analysis. This dataset provides a diverse range of images capturing live cell behavior and structures, enabling researchers to study and analyze cellular dynamics with precision.

### Dataset Structure
The dataset is organized into the following key components:
* Train:
  1. Images: This folder contains 3 raw microscopy images captured during live cell imaging experiments. Each image corresponds to a specific time frame and captures the intricate details of cell structures, cell divisions, and interactions.
  2. Labels: The label folder contains corresponding ground truth segmentations for the images. These label images define the regions of interest within the microscopy images, indicating cell boundaries, nuclei, and other cellular components.
  3. Borders: The border folder contains border information that aids in cell segmentation. Borders provide additional information about cell boundaries, helping the segmentation model accurately distinguish between adjacent cells.
* Test:
The test folder contains a separate set of microscopy images specifically reserved for testing the trained segmentation model. These images are not used during training or validation, ensuring an unbiased evaluation of the model's performance.

### Dataset Format
The images in the dataset are typically in TIFF format, which is a common format for microscopy images. Each image is represented as a matrix of pixel values, capturing grayscale information that corresponds to cell structures. Label images and border images are also in TIFF format, containing binary information that represents the segmented regions and cell borders. The size of the images in the LiveCell Imaging dataset is 1024x1024 pixels. 

### Dataset Preprocessing
Before training the U-Net model, the dataset undergoes several preprocessing steps, including:
- Normalization: Images are normalized to a standardized range, typically between 0 and 1, to ensure consistent input to the model.
- Resizing: Images are resized to a consistent resolution to match the input size of the U-Net model.
- Augmentation: Data augmentation techniques are applied to increase the diversity of the training dataset. This involves random rotations, flips, and other transformations.

## The Repository
In the repository, scientific Python programs have been implemented to complete the project steps.

**It consists of the following parts:**

/figures: This folder is the location where various plots and visualizations are saved. It contains images that visualize different stages of the data preprocessing, model training, testing, and other analysis steps. 

/model: This folder contains Python modules related to the neural network model used for image segmentation. It includes files defining the U-Net architecture, loss functions, training procedures, and prediction functions.

/plots: This is where contains Python scripts are responsible for creating various plots and visualizations used throughout the project. It includes functions to visualize images, labels, predictions, and other metrics.

/preprocess: This folder contains preprocessing-related modules. These include functions to load, preprocess, augment, and transform image and label data before training the model.

/unet_data: This folder is the main directory for storing the dataset and related files. It includes subdirectories for training, testing, and validation data, as well as augmented data. The predicted_masks subfolder contains the predicted 
segmentations obtained from the trained model.

/main.ipynb: This is the main Jupyter Notebook file where the entire pipeline of loading the data, preprocessing, building the U-Net model, training, testing, and evaluating the model is executed and documented.

/unet_model.pth: This file is the saved model weights of the trained U-Net model. The .pth extension indicates that it's a PyTorch model checkpoint file.

## Conclusion
The LiveCell Imaging dataset serves as a valuable resource for researchers and practitioners in the field of cell biology and microscopy. By leveraging the dataset and the U-Net segmentation model, insights into cellular behavior, interactions, and dynamics can be gained with a high degree of accuracy and efficiency. The dataset and accompanying pipeline offer a powerful tool for advancing our understanding of cellular processes.

## Acknowledgments
I would like to express my gratitude to Mr. Kouroudis, Ioannis for providing me access to the LiveCell Imaging dataset used in this project. The dataset can be accessed at [data source link](https://mega.nz/folder/G9hT3SRY#He6hD4SiU3g1bMFxgsbTDw). All lecturers' contribution has been instrumental in enabling me to carry out research and experiments in the field of image segmentation. I greatly appreciate their support and guidance throughout this project.

## Contributing
Contributions to this project are welcome! If you find any issues, have suggestions for improvements, or want to add new features, feel free to submit a pull request.


