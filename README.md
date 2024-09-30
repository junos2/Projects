# Python Research Projects

This repository showcases my work in Python, demonstrating applications of machine learning, dimensionality reduction, and mathematical modeling techniques. Below are detailed descriptions of three key projects highlighting my experience with supervised learning, neural networks, and scientific computing.

## 1. Supervised Learning Coursework: Regression & Classification Tasks

### Regression Task: Predicting Capacitance of Graphene-Based Supercapacitors
In this task, I predicted the electrical capacitance of graphene-based electrodes by leveraging supervised learning models. The design properties of these electrodes served as input features. The goal was to optimize model accuracy through the following techniques:
- **Decision Tree**
- **Random Forest**
- **Gradient Boosted Decision Trees (GBDT)**

Performance was assessed using **Mean Squared Error (MSE)** and **R² score**, with cross-validation and hyperparameter tuning applied to improve model robustness.

### Classification Task: Brain Cancer Diagnosis
I developed models to classify brain tumors (benign, glioma, meningioma) based on imaging data. Techniques employed included:
- **k-Nearest Neighbors (kNN)**
- **Weighted kNN**
- **Logistic Regression with Kernel Extension**

Models were evaluated using **accuracy**, **precision**, and **AUC-PR** scores, with cross-validation and test set evaluations to gauge generalization.

---

## 2. Stellar Classification and Dimensionality Reduction

This project uses star image data from the **Euclid telescope of the European Space Agency** to explore both image classification and dimensionality reduction.

### Convolutional Neural Network (CNN) for Image Classification
I implemented a CNN to classify stars into four spectral classes (O5V, A0V, F5V, M5V):
- Utilized **PyTorch** for model implementation.
- Applied techniques like **max-pooling**, **ReLU activation**, **Adam optimization**, and **categorical cross-entropy**.
- Mitigated class imbalance through **reweighted loss** and **data augmentation** strategies.

### Dimensionality Reduction
I applied dimensionality reduction techniques to visualize high-dimensional data:
- **Principal Component Analysis (PCA)** was used to reduce embeddings into a 2D space.
- An **Isomap-inspired algorithm** was implemented to construct a k-nearest neighbor graph, utilizing resistance distances for alternative 2D projections.

This project showcases the integration of CNNs with graph-based techniques, providing insights into both the classification and structural properties of the data.

---

## 3. Scientific Computing and Data Analysis

### Part 1: Zonal Wind Speed Analysis
Using a dataset of zonal wind speeds, I analyzed the daily and spatial fluctuations over latitude and longitude:
- Conducted **spectral analysis** using Discrete Fourier Transform (DFT) to uncover periodic patterns.
- Applied **PCA** to identify key factors influencing wind speed, with the first principal component explaining 55% of the variance.

### Part 2: Data Interpolation Methods
I implemented and compared two interpolation methods for data analysis:
- **Method 1**: Explicit method (average of adjacent values).
- **Method 2**: Implicit method using a higher-order interpolation with a banded matrix system.
  
Results indicated a trade-off between speed (Method 1) and accuracy (Method 2), with an error analysis of computational time and precision.

### Part 3: Simulation of Nonlinear ODEs
I solved a system of nonlinear ordinary differential equations (ODEs) and analyzed system behavior under different parameters:
- Used **solve_ivp** from SciPy for solving ODEs.
- Conducted **frequency analysis** using the Welch method to uncover chaotic behavior for higher parameter values.
- Applied **PCA** to the simulation data, highlighting how variance distribution changes with system parameters.


Part 3: Simulation of Nonlinear ODEs
I solved a system of nonlinear ordinary differential equations (ODEs) and analyzed system behavior under different parameters:

Used solve_ivp from SciPy for solving ODEs.
Conducted frequency analysis using the Welch method to uncover chaotic behavior for higher parameter values.
Applied PCA to the simulation data, highlighting how variance distribution changes with system parameters.
