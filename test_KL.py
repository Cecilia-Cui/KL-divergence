#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def calculate_kl_divergence(mu1, Sigma1, mu2, Sigma2):
    

    # Calculate determinants of the covariance matrices
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)

    # Calculate the inverse of the covariance matrices
    inv_Sigma2 = np.linalg.inv(Sigma2)

    # Dimensions of the data (assuming mu1 and mu2 are of the same dimension)
    d = mu1.shape[0]

    # Calculate the trace of the matrix product Sigma2^{-1} * Sigma1
    trace_term = np.trace(np.dot(inv_Sigma2, Sigma1))

    # Calculate the quadratic term ((mu2 - mu1)^T * Sigma2^{-1} * (mu2 - mu1))
    quadratic_term = np.dot(np.dot((mu2 - mu1).T, inv_Sigma2), (mu2 - mu1))

    # Calculate the KL divergence
    kl_divergence = 0.5 * (np.log(det_Sigma2/det_Sigma1) - d + trace_term + quadratic_term)

    return kl_divergence

# Example usage:
# Define the mean vectors and covariance matrices for two multivariate Gaussian distributions
mu1 = np.array([0, 0])
mu2 = np.array([1, 1])
Sigma1 = np.array([[1, 0.5], [0.5, 1]])
Sigma2 = np.array([[1, 0], [0, 1]])

# Calculate the KL divergence
kl_divergence = calculate_kl_divergence(mu1, Sigma1, mu2, Sigma2)
print(f"The KL divergence is: {kl_divergence}")