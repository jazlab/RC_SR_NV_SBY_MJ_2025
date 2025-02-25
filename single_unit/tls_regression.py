"""
This module contains a function that performs total least squares regression.
"""

import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt


def tls_regression(x, y):
    # Define the model function for the regression line (linear in this case)
    def linear_model(B, x):
        return B[0] * x + B[1]

    # Function to calculate orthogonal distances from points to a line
    def orthogonal_distances(x, y, B):
        m, c = B
        distances = np.abs(m * x - y + c) / np.sqrt(m**2 + 1)
        return np.sum(distances**2)

    # Create a Model object and RealData object
    model = Model(linear_model)
    data = RealData(x, y)

    # Set up ODR with the model and data
    odr = ODR(data, model, beta0=[1.0, 2.0])

    # Run the regression
    out = odr.run()

    # Calculate the total and residual orthogonal variances
    x_mean, y_mean = np.mean(x), np.mean(y)
    total_variance = orthogonal_distances(x, y, [0, y_mean])
    residual_variance = orthogonal_distances(x, y, out.beta)
    explained_variance = total_variance - residual_variance
    explained_variance_ratio = explained_variance / total_variance

    # Prepare the plot
    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = linear_model(out.beta, x_plot)

    fig = plt.figure(figsize=(2, 2))
    plt.scatter(x, y, color="black", label="")
    plt.plot(
        x_plot,
        y_plot,
        color="blue",
        label=f"Slope: {out.beta[0]:.2f}\nExplained Variance Ratio: {explained_variance_ratio:.2f}",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Total Least Squares Regression with Explained Variance")
    plt.legend()
    return fig, out.beta, out.sd_beta, explained_variance_ratio
