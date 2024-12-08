# residual_analysis.py

import matplotlib.pyplot as plt
import numpy as np

def plot_residuals(x, y, slope, y_intercept):
    """
    Plots the residuals (differences between observed and predicted values) against the predicted values.
    
    Parameters:
        x (list or tuple): Independent variable values
        y (list or tuple): True dependent variable values
        slope (float): Slope of the fitted regression line
        y_intercept (float): Y-intercept of the fitted regression line
    """
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("x and y must be iterables")
    
    # Calculate predicted y values
    predicted_y = [(slope * xi) + y_intercept for xi in x]
    print(f"Predicted y: {predicted_y}")  # Debug print
    
    # Calculate residuals
    residuals = [yi - predicted for yi, predicted in zip(y, predicted_y)]
    print(f"Residuals: {residuals}")  # Debug print
    
    # Plot residuals vs. predicted y values
    plt.scatter(predicted_y, residuals)
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y = 0
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()  # Ensure the plot is displayed

    return residuals


def plot_residual_histogram(x, y, slope, y_intercept):
    """
    Plots a histogram of the residuals to check for normality.
    
    Parameters:
        x (list or tuple): Independent variable values
        y (list or tuple): True dependent variable values
        slope (float): Slope of the fitted regression line
        y_intercept (float): Y-intercept of the fitted regression line
    """
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("x and y must be iterables")

    # Calculate predicted y values
    predicted_y = [(slope * xi) + y_intercept for xi in x]
    print(f"Predicted y: {predicted_y}")  # Debug print
    
    # Calculate residuals
    residuals = [yi - predicted for yi, predicted in zip(y, predicted_y)]
    print(f"Residuals: {residuals}")  # Debug print
    
    # Plot histogram of residuals
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residuals Histogram")
    plt.show()  # Ensure the plot is displayed

    return residuals
