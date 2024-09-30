# Python Research Projects

This repository showcases my work in Python, demonstrating applications of machine learning, dimensionality reduction, and mathematical modeling techniques. Below are detailed descriptions of three key projects highlighting my experience with supervised learning, neural networks, and scientific computing.

## 1. Supervised Learning: Regression & Classification Tasks

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

---

## 4. Time Series Analysis and AR Processes

This repository contains code and documentation for a personal research project focused on time series analysis, particularly utilizing Autoregressive (AR) models. The work includes simulating AR processes, spectral estimation, and forecasting sea level data using real-world datasets.

### Project Overview
This research investigates the properties of AR processes, emphasizing the simulation of an AR(4) process, estimating spectral densities, and forecasting using fitted AR models. It encompasses analyses based on synthetic data as well as real-world sea level gauge data.

### Research Components
#### AR Processes Simulation and Spectral Estimation
- Developed a function to simulate an AR(4) process that oscillates with two dominant frequencies. The function accepts frequency inputs and coefficients controlling cyclical behaviors. A burn-in method was applied to ensure a stationary process by discarding initial values.
- Created a function to compute the theoretical spectral density of an AR(p) process based on given frequencies and parameters.
- Implemented functions to compute the periodogram and direct spectral estimates using cosine tapers at various values, allowing for the examination of bias in spectral estimators.

#### Spectral Density Function Estimation
- Applied a direct spectral estimator to a 10-year monthly sea level gauge dataset from the National Oceanographic Centre. Centered the data to remove the mean, which significantly affected spectral density estimation.

#### Forecasting and Prediction Intervals
- Fitted AR(p) models using Yule-Walker and maximum likelihood estimation methods. Employed a rolling origin approach to determine the best model order based on root mean square error (RMSE).
- Used the selected AR model to forecast the next 12 months of sea level data and computed prediction intervals for the forecasts.
