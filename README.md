# Plant Disease Predictor 🌿
A deep learning model built with TensorFlow/Keras to classify plant diseases from leaf images. This project uses a Convolutional Neural Network (CNN) to identify 38 different classes of plant diseases from the PlantVillage dataset.

## 📋 Table of Contents
Overview

Features

Demo

Technology Stack

Setup and Installation

Usage

Model Architecture

Results

Future Improvements

### 📝 Overview
This project aims to accurately identify various diseases in plants by analyzing images of their leaves. The model is trained on the PlantVillage dataset, which contains over 54,000 images of healthy and diseased plant leaves, categorized into 38 distinct classes. The entire workflow, from data acquisition to model training and prediction, is documented in the provided Jupyter Notebook.

### ✨ Features
Image Classification: Classifies plant leaf images into 38 different disease categories.

Data Augmentation: Utilizes ImageDataGenerator for efficient data loading and real-time data augmentation.

CNN Architecture: A simple yet effective custom CNN model built from scratch.

Prediction Script: Includes a function to easily load a new image and predict its class.

Saved Model: The trained model and class indices are saved for easy deployment and use.

🛠️ Technology StackPython: Core programming language.TensorFlow & Keras: For building and training the deep learning model.NumPy: For numerical operations.Matplotlib: For plotting training history and visualizing images.Pillow (PIL): For image manipulation.Kaggle API: For downloading the dataset.Jupyter Notebook: For interactive development and documentation.⚙️ Setup and InstallationFollow these steps to set up the project on your local machine.1. PrerequisitesPython 3.9+pip package manager2. Clone the RepositoryBashgit clone https://github.com/your-username/plant-disease-predictor.git
cd plant-disease-predictor
3. Install DependenciesCreate a requirements.txt file with the following content:Plaintextnumpy
tensorflow
matplotlib
Pillow
kaggle
scikit-learn
Then, install the required packages:Bashpip install -r requirements.txt
4. Kaggle API SetupCreate a Kaggle account and download your API token (kaggle.json).Place the kaggle.json file in the root directory of this project.5. Download the DatasetRun the initial cells in the Jupyter notebook (Plant_Disease_Prediction.ipynb) to automatically download and unzip the PlantVillage dataset using the Kaggle API.UsageThe primary file is the Jupyter Notebook (Plant_Disease_Prediction.ipynb), which contains the complete workflow.Open the Notebook: Launch Jupyter Notebook and open Plant_Disease_Prediction.ipynb.Run All Cells: You can run all the cells sequentially to perform the following actions:Download and extract the dataset.Preprocess the data.Build, compile, and train the CNN model.Evaluate the model and visualize the results.Save the trained model as plant_disease_prediction_model.h5 and class indices as class_indices.json.Predict on a New Image:Place your test image in the root directory.Modify the image_path variable in the final cells of the notebook.Run the prediction cell to see the classification result.Python# Example of making a prediction
image_path = 'path/to/your/test_image.jpg'
predicted_class = predict_image_class(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class)
🧠 Model ArchitectureThe project uses a custom Convolutional Neural Network (CNN). The architecture is defined as follows:Convolutional Layer: 32 filters, (3, 3) kernel, ReLU activation.Max Pooling Layer: (2, 2) pool size.Convolutional Layer: 64 filters, (3, 3) kernel, ReLU activation.Max Pooling Layer: (2, 2) pool size.Flatten Layer: To convert 2D feature maps into a 1D vector.Dense Layer: 256 neurons, ReLU activation.Output Layer: num_classes neurons (38), softmax activation for multi-class classification.📊 ResultsThe model was trained for 5 epochs and achieved the following performance on the validation set:Validation Accuracy: XX.XX% (You should replace this with your final accuracy, e.g., 88.32%)Training HistoryThe plots below show the model's accuracy and loss over the training epochs.Model AccuracyModel Loss(To use this, save your plots as images and add them to your repository)💡 Future ImprovementsTransfer Learning: Implement a pre-trained model (like MobileNetV2 or ResNet50) to potentially achieve higher accuracy with faster training.Enhanced Data Augmentation: Apply more advanced augmentation techniques (rotation, zoom, shear) to improve model generalization.Deployment: Deploy the trained model using a web framework like Flask or Streamlit to create an interactive web application.Fine-tuning: Unfreeze some layers of a pre-trained model and fine-tune them on the PlantVillage dataset for even better performance.
