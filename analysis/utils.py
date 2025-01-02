import numpy as np
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