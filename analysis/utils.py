import numpy as np
from scipy.stats import t

def dice_coefficient(matrix1, matrix2):
    """
    Calculate the Dice coefficient between two binary matrices.
    """
    # Ensure the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("The matrices must have the same shape.")
    
    # Flatten the matrices (optional, for clarity)
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()
    
    # Calculate intersection and sums
    intersection = np.sum(matrix1 * matrix2)
    sum1 = np.sum(matrix1)
    sum2 = np.sum(matrix2)
    
    # Handle the case where both matrices are empty
    if sum1 + sum2 == 0:
        return 1.0  # Define Dice coefficient as 1 for empty matrices
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (sum1 + sum2)
    return dice

def compute_t_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for the mean of a given array using the t-distribution.
    
    Parameters:
        data (numpy.ndarray): Array of numerical data.
        confidence (float): Confidence level (default is 0.95 for 95% confidence interval).
    
    Returns:
        tuple: (mean, lower_bound, upper_bound) of the confidence interval.
    """
    if len(data) == 0:
        raise ValueError("The input array is empty.")
    
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    n = len(data)
    
    # Compute the t-critical value for the given confidence level and degrees of freedom
    t_crit = t.ppf(1 - (1 - confidence) / 2, df=n-1)
    
    # Compute margin of error
    margin_of_error = t_crit * (std_dev / np.sqrt(n))
    
    # Compute the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return mean, lower_bound, upper_bound