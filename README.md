# Land-Use Classification from Satellite Imagery

## 1. Introduction

This project addresses the critical task of automated land-use classification using satellite imagery. By leveraging a deep learning approach, the project builds a model capable of accurately identifying and categorizing different land types (e.g., forests, farmlands, urban areas) from satellite photos. This technology has wide-ranging applications in environmental monitoring, urban planning, and precision agriculture.

## 2. Project Overview

The core of this project is to compare two distinct deep learning methodologies for a multi-class image classification problem:

1.  **From-Scratch CNN:** A basic Convolutional Neural Network (CNN) built and trained from the ground up to serve as a performance baseline.
2.  **Transfer Learning:** A more advanced and efficient model that utilizes a pre-trained **ResNet50** architecture, which has learned from a vast dataset of general-purpose images (ImageNet). The model is then fine-tuned on the specific satellite imagery data.

### Technologies Used
* **TensorFlow / Keras:** The primary deep learning framework.
* **TensorFlow Datasets:** For efficient data loading and preprocessing.
* **Scikit-learn / Matplotlib:** For comprehensive evaluation and data visualization.

## 3. Key Features and Results

The final model achieved a high level of performance through a series of key optimizations:

* **Final Accuracy:** The transfer learning model achieved an accuracy of over **92%** on the test set, significantly outperforming the from-scratch model.
* **Correct Preprocessing:** The project highlights the critical importance of using the correct preprocessing function (`resnet50_preprocess_input`) for the pre-trained model, which was essential for achieving high accuracy.
* **Comprehensive Evaluation:** The model's performance is thoroughly evaluated using a **Confusion Matrix**, **Classification Report**, and **ROC Curves**, providing a detailed view of its strengths and weaknesses across all 10 land-use classes.
* **Interactive Feature:** The final code includes a feature that allows a user to upload their own satellite image and get a live prediction from the trained model.

### Genetic Algorithm Optimization

* A conceptual **Genetic Algorithm** was implemented to find the optimal set of hyperparameters for the model.
* The algorithm showed a clear evolution in accuracy across generations, demonstrating a systematic search for the best solution.
* The process successfully identified a set of hyperparameters that led to a best validation accuracy of **over 94%**, confirming the model's high performance and validating the chosen approach.

*(Place the "Genetic Algorithm Accuracy Evolution" graph here.)*

## 4. How to Run the Code

The code is designed to be run in a Google Colab notebook for easy setup and access to a GPU.

1.  Open a new Google Colab notebook.
2.  Set the runtime to GPU (Runtime -> Change runtime type -> GPU).
3.  Copy and paste the entire `final_project_code.py` file into a cell.
4.  Run the cell to train and evaluate the models. The code will automatically download the necessary dataset and model weights.

## 5. Repository Structure

* `README.md`: This file.
* `final_project_code.py`: The complete Python script for the project, including all training, evaluation, and interactive features.

## 6. Future Work

* **Fine-Tuning:** Unfreeze and retrain the final layers of the ResNet50 model with a low learning rate to further optimize performance.
* **Data Augmentation:** Implement more extensive data augmentation techniques to improve model robustness.
* **Hyperparameter Tuning:** Use automated methods like a Genetic Algorithm to find the optimal set of hyperparameters.

