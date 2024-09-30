# -*- coding: utf-8 -*-
"""
Code for Time Series Analysis and AR Processes

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Part 1: Simulate AR(4) Process
# -------------------------------

def simulate_AR4(r1, f1, r2, f2, num_samples, burn_in=1000):
    """
    Simulates an AR(4) process.

    Parameters:
    - r1, r2: AR coefficients
    - f1, f2: Frequencies
    - num_samples: Total number of samples to simulate
    - burn_in: Number of initial samples to discard

    Returns:
    - ar_process: Simulated AR(4) process after burn-in
    """
    # Calculate the complex roots
    a = r1 * np.exp(1j * 2 * np.pi * f1)
    b = r1 * np.exp(-1j * 2 * np.pi * f1)
    c = r2 * np.exp(1j * 2 * np.pi * f2)
    d = r2 * np.exp(-1j * 2 * np.pi * f2)

    # Calculate AR coefficients
    phi1 = a + b + c + d
    phi2 = -((a + b) * (c + d) + a * b + c * d)
    phi3 = a * b * (c + d) + c * d * (a + b)
    phi4 = -a * b * c * d

    # Simulate the AR(4) process with burn-in
    ar_process = np.zeros(num_samples + burn_in)
    for t in range(4, num_samples + burn_in):
        ar_process[t] = (phi1 * ar_process[t-1] +
                         phi2 * ar_process[t-2] +
                         phi3 * ar_process[t-3] +
                         phi4 * ar_process[t-4] +
                         np.random.normal())

    # Discard the burn-in samples
    return ar_process[burn_in:]

# -------------------------------
# Part 2: Spectral Density Functions
# -------------------------------

def S_AR(frequencies, phis, sigma2):
    """
    Computes the spectral density function for AR(p) process.

    Parameters:
    - frequencies: Frequencies to compute the spectral density
    - phis: AR coefficients
    - sigma2: Variance of the process

    Returns:
    - spectral_density: Spectral density at given frequencies
    """
    p = len(phis)
    spectral_density = np.zeros_like(frequencies)

    # Calculate the spectral density
    for i, f in enumerate(frequencies):
        sum_term = np.sum([phis[j] * np.exp(-1j * 2 * np.pi * (j + 1) * f) for j in range(p)])
        denominator = np.abs(1 - sum_term) ** 2
        spectral_density[i] = sigma2 / denominator

    return spectral_density

def periodogram(X):
    """
    Computes the periodogram of a time series.

    Parameters:
    - X: Time series data

    Returns:
    - f: Frequencies
    - Pxx: Periodogram estimate
    """
    N = len(X)
    X_fft = np.fft.fft(X)
    Pxx = np.abs(X_fft) ** 2 / N
    f = np.fft.fftfreq(N)
    return np.fft.fftshift(f), np.fft.fftshift(Pxx)

def direct(X, p):
    """
    Computes the direct spectral estimate using tapering.

    Parameters:
    - X: Time series data
    - p: Taper percentage

    Returns:
    - f: Frequencies
    - spec_est: Spectral estimate
    """
    N = len(X)
    k = int(p * N)
    taper = np.zeros(N)

    # Create taper
    for t in range(1, N + 1):
        if 1 <= t <= k / 2:
            taper[t - 1] = 1/2 * (1 - np.cos(2 * np.pi * t / (k + 1)))
        elif k / 2 < t <= N - k / 2:
            taper[t - 1] = 1
        elif N + 1 - k / 2 < t <= N:
            taper[t - 1] = 1/2 * (1 - np.cos(2 * np.pi * (N + 1 - t) / (k + 1)))

    # Normalize taper
    taper /= np.linalg.norm(taper)
    X_tapered = X * taper
    spec_est = np.fft.fftshift(np.fft.fft(X_tapered))
    f = np.fft.fftfreq(N)

    return np.fft.fftshift(f), np.abs(spec_est) ** 2

# -------------------------------
# Part 3: Simulation and Bias Calculation
# -------------------------------

# Initialize values for simulations
num_realizations = 5000
N = 64
sigma2 = 1
frequencies_of_interest = [6/64, 8/64, 16/64, 26/64]
r1 = r2 = 0.8
f1 = 6/64
f2 = 26/64
p_values = [0.05, 0.1, 0.25, 0.5]

# Arrays to store results
periodogram_results = np.zeros((num_realizations, N))
direct_results = {p: np.zeros((num_realizations, N)) for p in p_values}

# Run simulations
for i in range(num_realizations):
    ar_process = simulate_AR4(r1, f1, r2, f2, N)

    # Compute periodogram
    freq, periodogram_result = periodogram(ar_process)
    periodogram_results[i, :] = periodogram_result

    # Compute direct spectral estimates for different taper percentages
    for p in p_values:
        freq_direct, direct_result = direct(ar_process, p)
        direct_results[p][i, :] = direct_result

# Coefficients for the AR(4) process
a = r1 * np.exp(1j * 2 * np.pi * f1)
b = r1 * np.exp(-1j * 2 * np.pi * f1)
c = r2 * np.exp(1j * 2 * np.pi * f2)
d = r2 * np.exp(-1j * 2 * np.pi * f2)

phi1 = a + b + c + d
phi2 = -((a + b) * (c + d) + a * b + c * d)
phi3 = a * b * (c + d) + c * d * (a + b)
phi4 = -a * b * c * d

# Calculate bias
bias_periodogram = []
bias_direct = {p: [] for p in p_values}
freq_list = freq.tolist()

for f in frequencies_of_interest:
    true_spectral_density = S_AR([f], [phi1, phi2, phi3, phi4], sigma2)[0]
    f_i = freq_list.index(f)
    mean_periodogram = np.mean(periodogram_results[:, f_i])
    bias_periodogram.append((mean_periodogram - np.real(true_spectral_density)) / np.real(true_spectral_density) * 100)

for p in p_values:
    for f in frequencies_of_interest:
        f_i = freq_list.index(f)
        true_spectral_density = S_AR([f], [phi1, phi2, phi3, phi4], sigma2)[0]
        mean_direct = np.mean(direct_results[p][:, f_i])
        bias_direct[p].append((mean_direct - np.real(true_spectral_density)) / np.real(true_spectral_density) * 100)

# -------------------------------
# Part 4: Analyze Bias with Varying Parameters
# -------------------------------

r_values = np.linspace(0.8, 0.99, 20)
bias_per_dict = {f: [] for f in frequencies_of_interest}

for f in frequencies_of_interest:
    for r in r_values:
        num_realizations = 500
        N = 64
        sigma2 = 1
        r1 = r2 = r
        f1 = 6/64
        f2 = 26/64

        # Arrays to store results
        periodogram_results = np.zeros((num_realizations, N))
        direct_results = {p: np.zeros((num_realizations, N)) for p in p_values}

        for i in range(num_realizations):
            ar_process = simulate_AR4(r1, f1, r2, f2, N)

            # Compute periodogram
            freq, periodogram_result = periodogram(ar_process)
            periodogram_results[i, :] = periodogram_result

            # Compute direct spectral estimates for different taper percentages
            for p in p_values:
                freq_direct, direct_result = direct(ar_process, p)
                direct_results[p][i, :] = direct_result

        # Coefficients for the AR(4) process
        a = r1 * np.exp(1j * 2 * np.pi * f1)
        b = r1 * np.exp(-1j * 2 * np.pi * f1)
        c = r2 * np.exp(1j * 2 * np.pi * f2)
        d = r2 * np.exp(-1j * 2 * np.pi * f2)

        # Calculate bias for periodogram
        true_spectral_density = S_AR([f], [phi1, phi2, phi3, phi4], sigma2)[0]
        f_i = freq.tolist().index(f)
        mean_periodogram = np.mean(periodogram_results[:, f_i])
        bias_periodogram = (mean_periodogram - np.real(true_spectral_density)) / np.real(true_spectral_density) * 100
        bias_per_dict[f].append(bias_periodogram)

# -------------------------------
# Part 5: Yule-Walker Equations
# -------------------------------

def yw(X, p):
    """
    Estimates AR parameters using Yule-Walker equations.

    Parameters:
    - X: Time series data
    - p: Order of the AR process

    Returns:
    - phis: Estimated AR coefficients
    """
    r = np.correlate(X - np.mean(X), X - np.mean(X), mode='full')[len(X)-1:]
    R = np.array([r[i:i + p] for i in range(p)])
    R = np.linalg.inv(R)
    a = np.array([-np.dot(R[i], r[i + 1:i + p + 1]) for i in range(p)])
    return a

# -------------------------------
# Part 6: Forecasting
# -------------------------------

def forecast(X, phis, n_steps):
    """
    Makes forecasts based on AR coefficients.

    Parameters:
    - X: Time series data
    - phis: AR coefficients
    - n_steps: Number of steps to forecast

    Returns:
    - forecasts: Forecasted values
    """
    forecasts = []
    for _ in range(n_steps):
        forecast_value = sum(phis[j] * X[-(j + 1)] for j in range(len(phis)))
        forecasts.append(forecast_value)
        X = np.append(X, forecast_value)
    return forecasts

# -------------------------------
# Part 7: RMSE Calculation
# -------------------------------

def rmse(actual, predicted):
    """
    Calculates the Root Mean Square Error (RMSE).

    Parameters:
    - actual: Actual values
    - predicted: Predicted values

    Returns:
    - rmse_value: Calculated RMSE
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))

# -------------------------------
# Part 8: Load Data and Preprocess
# -------------------------------

def load_data(file_path):
    """
    Loads time series data from an Excel file.

    Parameters:
    - file_path: Path to the Excel file

    Returns:
    - data: Loaded time series data
    """
    return pd.read_excel(file_path)

# Example usage:
# data = load_data('path_to_excel_file.xlsx')
# ar_process = simulate_AR4(0.8, 0.1, 0.5, 0.2, 1000)
# forecasts = forecast(ar_process, [0.5, 0.2, -0.1, 0.3], n_steps=10)
