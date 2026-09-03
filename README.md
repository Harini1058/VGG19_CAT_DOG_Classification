# VGG19_CAT_DOG_Classification
# 🔍 Defect Detection using Transfer Learning

A deep learning image classification project that uses **Transfer Learning with VGG19** to classify images into two categories: **Defect** and **Not Defect**.

## 👩‍💻 Author

**Harini G**
**Register Number:** 212225230091
**Slot:** 26OD1143

---

## 📌 Project Overview

This project develops a neural network classification model using **transfer learning** with the pre-trained **VGG19** architecture.

The model is trained to identify whether an input image belongs to one of two classes:

* 🔴 **Defect**
* 🟢 **Not Defect**

Instead of training a deep CNN completely from scratch, the pre-trained VGG19 feature extraction layers are used to extract meaningful visual features. The feature extraction layers are frozen, and a new classifier is trained for the target classification task.

---

## 🎯 Objectives

* Implement image classification using PyTorch.
* Apply transfer learning using the VGG19 architecture.
* Extract meaningful features from input images.
* Classify images into defect and non-defect categories.
* Evaluate the model using accuracy, precision, recall, and F1-score.
* Visualize classification performance using a confusion matrix.
* Perform prediction on individual images.

---

## 📊 Dataset

The dataset contains images organized into training and testing directories and is loaded using PyTorch's `ImageFolder` dataset loader.

### Classes

| Class       | Description               |
| ----------- | ------------------------- |
| `defect`    | Images containing defects |
| `notdefect` | Images without defects    |

The project contains:

| Dataset  | Number of Images |
| -------- | ---------------: |
| Training |              172 |
| Testing  |              121 |
| Total    |              293 |

## The images are resized to **224 × 224 pixels** and converted into tensors before being passed to the model.

## 🧠 Model Architecture

The project uses **VGG19**, a pre-trained convolutional neural network available through Torchvision.

### Model Pipeline

```text
Input Image
     │
     ▼
Resize to 224 × 224
     │
     ▼
Pre-trained VGG19
     │
     ▼
Feature Extraction Layers
     │
     ▼
Average Pooling
     │
     ▼
Flatten Features
     │
     ▼
Fully Connected Classifier
     │
     ├──────────────┐
     ▼              ▼
  Defect        Not Defect
```

The original final classification layer of VGG19 is replaced with a new linear layer containing **2 output classes**.

---

## 🔄 Transfer Learning Approach

The pre-trained VGG19 model is used as a feature extractor.

The feature extraction layers are frozen:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

This prevents the pre-trained convolutional layers from being updated during training.

A separate classifier is then trained using the extracted VGG19 features.

The extracted feature dimensions were:

```text
Training features: torch.Size([172, 25088])
Testing features : torch.Size([121, 25088])
```

---

## ⚙️ Training Configuration

| Parameter         | Value             |
| ----------------- | ----------------- |
| Framework         | PyTorch           |
| Model             | VGG19             |
| Learning Method   | Transfer Learning |
| Input Size        | 224 × 224         |
| Number of Classes | 2                 |
| Batch Size        | 32                |
| Optimizer         | Adam              |
| Learning Rate     | 0.001             |
| Loss Function     | CrossEntropyLoss  |
| Classifier Epochs | 5                 |
| Device            | CPU               |

The project automatically checks whether CUDA is available and selects the appropriate device. In the recorded execution, the model was run on **CPU**.

---

## 📈 Training Results

The classifier was trained for 5 epochs.

| Epoch |   Loss |
| ----: | -----: |
|     1 | 0.7360 |
|     2 | 1.0266 |
|     3 | 0.6023 |
|     4 | 0.1868 |
|     5 | 0.1674 |

The final training loss was **0.1674**.

---

## 🏆 Model Performance

The trained classifier achieved:

# **90.08% Test Accuracy**

```text
Test Accuracy: 0.9008
Test Accuracy: 90.08%
```

The model correctly classified the majority of the 121 test images.

---

## 📋 Classification Report

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| Defect           |      0.73 |   1.00 |     0.85 |      33 |
| Not Defect       |      1.00 |   0.86 |     0.93 |      88 |
| **Accuracy**     |           |        | **0.90** | **121** |
| Macro Average    |      0.87 |   0.93 |     0.89 |     121 |
| Weighted Average |      0.93 |   0.90 |     0.90 |     121 |

The **Defect** class achieved a recall of **1.00**, meaning all defect samples in the test set were identified by the classifier.

---

## 📊 Confusion Matrix

A confusion matrix is generated using Scikit-learn to analyze the predictions for both classes.

```python
cm = confusion_matrix(
    test_labels.cpu().numpy(),
    predicted.cpu().numpy()
)
```

The matrix provides a visual comparison between the **actual classes** and the **predicted classes**.

---

## 🔮 Single Image Prediction

The project also includes functionality to classify an individual test image.

Example prediction:

```text
Actual: notdefect
Predicted: notdefect
```

The model successfully classified the sample image as **Not Defect**.

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **VGG19**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Seaborn**
* **Torchsummary**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/defect-detection-transfer-learning.git
```

Navigate to the project:

```bash
cd defect-detection-transfer-learning
```

Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn torchsummary
```

---

## 📁 Project Structure

```text
Defect-Detection-Transfer-Learning/
│
├── data/
│   └── dataset/
│       ├── train/
│       │   ├── defect/
│       │   └── notdefect/
│       │
│       └── test/
│           ├── defect/
│           └── notdefect/
│
├── EX04_Transfer_Learning.ipynb
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Prepare the Dataset

Place the dataset in the following structure:

```text
data/
└── dataset/
    ├── train/
    │   ├── defect/
    │   └── notdefect/
    │
    └── test/
        ├── defect/
        └── notdefect/
```

### 2. Run the Notebook

Open:

```text
EX04_Transfer_Learning.ipynb
```

Execute the cells sequentially.

### 3. The Program Will

1. Import the required libraries.
2. Preprocess the images.
3. Load the training and testing datasets.
4. Display sample images.
5. Create DataLoaders.
6. Load the pre-trained VGG19 model.
7. Modify the final classification layer.
8. Freeze the VGG19 feature extraction layers.
9. Extract image features.
10. Train the classifier.
11. Evaluate the model.
12. Generate a confusion matrix.
13. Generate a classification report.
14. Predict an individual image.

---

## 💡 Key Learning Outcomes

This project provides practical experience with:

* Transfer learning
* Pre-trained CNN architectures
* VGG19
* Feature extraction
* Image preprocessing
* PyTorch `ImageFolder`
* DataLoaders
* Model freezing
* Neural network classification
* Cross-entropy loss
* Adam optimization
* Confusion matrix analysis
* Classification reports
* Single-image inference

---

## 🚀 Future Improvements

The project can be further improved by:

* Increasing the size of the training dataset.
* Applying image augmentation.
* Using VGG19 fine-tuning instead of only feature extraction.
* Comparing VGG19 with ResNet, EfficientNet, or MobileNet.
* Adding validation data during training.
* Plotting training and validation accuracy/loss.
* Using GPU acceleration for faster training.
* Deploying the model as a web application.
* Adding real-time defect detection.

---

## 📌 Conclusion

This project demonstrates the application of **transfer learning for binary image classification** using a pre-trained VGG19 network.

By freezing the pre-trained feature extraction layers and training a dedicated classifier, the model achieved a **90.08% test accuracy** in distinguishing between **defect** and **notdefect** images.

The project provides a complete workflow from dataset preparation and feature extraction to model evaluation and individual image prediction.

---

## 📊 Project Summary

```text
Project        : Defect Detection using Transfer Learning
Model          : VGG19
Learning       : Transfer Learning
Classes        : 2
Training Data  : 172 images
Testing Data   : 121 images
Input Size     : 224 × 224
Epochs         : 5
Optimizer      : Adam
Learning Rate  : 0.001
Test Accuracy  : 90.08%
Device         : CPU
```

---

## ⭐ Acknowledgement

This project was developed as part of an academic deep learning exercise focused on **Neural Network Classification using Transfer Learning**.
