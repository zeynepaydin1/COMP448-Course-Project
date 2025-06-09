# Brain Tumor MRI Classification

COMP 448 Semester Project — Koç University  
**Authors**: Zeynep Aydın (76687), Hüseyin Sarp Vulaş (76602)  
**Date**: June 2025

## Overview

This project compares traditional handcrafted pipelines with deep convolutional neural networks for classifying brain tumor MRI slices. Using a clean and reproducible 5-fold cross-validation setup, we evaluate accuracy, robustness, and training efficiency across different approaches.

The dataset includes over 3000 labeled T1-weighted MRI slices (meningioma, glioma, pituitary) sourced from the public Figshare repository.  
All experiments were conducted on Google Colab with NVIDIA L4 GPU acceleration for deep learning models.

## Project Structure

📁 comp448project/
│
├── comp448project.ipynb # Main Colab notebook with full pipeline
├── report.pdf # Final LaTeX report (compiled)
├── report.tex # LaTeX source code
├── figures/ # Screenshots, confusion matrices, and plots
│
└── README.md # This file

## Methods

### 1. Traditional ML Pipelines
- **GLCM + SVM**
- **LBP + Random Forest**
- **Shape & Intensity + Logistic Regression**
- **HOG + KNN**

### 2. Deep Learning Models
- **SimpleCNN** (trained from scratch)
- **ResNet18 & DenseNet121** (transfer learning from ImageNet)

All deep models are trained using PyTorch with early preprocessing via `torchvision.transforms`. Augmentations include random rotation and horizontal flips.

## Results Summary

| Model            | Accuracy (5-fold CV) |
|------------------|----------------------|
| HOG + KNN        | 96.4%                |
| SimpleCNN        | 85.3%                |
| ResNet18 (TL)    | 96.5%                |
| DenseNet121 (TL) | **97.8%**            |

- HOG + KNN performs surprisingly well without GPU support.
- DenseNet121 provides the highest accuracy with minimal training time (3 epochs/fold).

## Requirements

- Python 3.10+
- Google Colab environment with PyTorch support
- Packages: `scikit-learn`, `scikit-image`, `matplotlib`, `numpy`, `torch`, `torchvision`, `h5py`

## How to Run

1. Open `comp448project.ipynb` in [Google Colab](https://colab.research.google.com).
2. Mount your Google Drive when prompted.
3. Place the `brain_tumor_dataset` folder containing `.mat` files into your drive.
4. Run all cells sequentially to preprocess, extract features, train models, and generate evaluation metrics and figures.

## Dataset

- **Source**: [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- 3064 slices, each with a raw MRI image, binary tumor mask, and label (1: Meningioma, 2: Glioma, 3: Pituitary)

## License

This code is released for educational purposes under the MIT License.

## Acknowledgements

- Figshare dataset authors
- Koç University COMP 448 instructors
- PyTorch and Scikit-learn development teams

