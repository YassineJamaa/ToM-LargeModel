import numpy as np
from src import LocImportantUnits

def get_masked_ktop(loc: LocImportantUnits, percentage: float):
    if percentage==0:
        return np.zeros_like(loc.t_values, dtype=np.int64)
    num_top_elements = int(loc.t_values.size * percentage)

    # Flatten the matrix, find the threshold value for the top 1%
    flattened_matrix = loc.t_values.flatten()

    # Replace NaN with a sentinel value (e.g., -inf)
    flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)

    # Filter out -inf values
    filtered_matrix = flattened_matrix[flattened_matrix != -np.inf]
    if len(filtered_matrix) > num_top_elements:
        threshold_value = np.partition(flattened_matrix, -num_top_elements)[-num_top_elements]
    else:
        threshold_value = -np.inf  # or handle this case as needed      

    # Create a binary mask where 1 represents the top 1% elements, and 0 otherwise
    mask_units = np.where(loc.t_values >= threshold_value, 1, 0)
    return mask_units

def get_masked_kbot(loc: LocImportantUnits, percentage):
    if percentage==0:
        return np.zeros_like(loc.t_values, dtype=np.int64)
    num_top_elements = int(loc.t_values.size * percentage)

    # Flatten the matrix, find the threshold value for the top 1%
    flattened_matrix = loc.t_values.flatten()

    # Replace NaN with a sentinel value (e.g., -inf)
    flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)
    
    # Filter out -inf values
    filtered_matrix = flattened_matrix[flattened_matrix != -np.inf]
    if len(filtered_matrix) > num_top_elements:
        threshold_value = np.partition(filtered_matrix, num_top_elements)[num_top_elements]
    else:
        threshold_value = -np.inf  # or handle this case as needed      
    # Create a binary mask where 1 represents the top 1% elements, and 0 otherwise
    mask_units = np.where(loc.t_values <= threshold_value, 1, 0)
    return mask_units

def get_masked_middle(loc: LocImportantUnits, percentage):
    # Flatten T-values
    t_units = loc.t_values.flatten()
    num_top_elements = int(loc.t_values.size * percentage)
    flattened_matrix = loc.t_values.flatten()
    flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)

    # Filter out -inf values
    filtered_matrix = flattened_matrix[flattened_matrix != -np.inf]

    # Calculate the median or mean of the filtered matrix
    value = np.mean(filtered_matrix)  # or use np.median(filtered_matrix)

    # Calculate the absolute deviations from the median/mean
    abs_deviations = np.abs(filtered_matrix - value)

    # Sort the absolute deviations and find the threshold
    if len(filtered_matrix) > num_top_elements:
        threshold_deviation = np.partition(abs_deviations, num_top_elements)[num_top_elements]
        threshold_value_left = value - threshold_deviation
        threshold_value_right = value + threshold_deviation
    else:
        threshold_value_left = -np.inf
        threshold_value_right = np.inf

    # Create a binary mask where 1 represents values within the middle percentage range, and 0 otherwise
    mask_units = np.where(
        (loc.t_values >= threshold_value_left) & (loc.t_values <= threshold_value_right), 1, 0
    )
    return mask_units


def get_masked_random(loc: LocImportantUnits, percentage: float, seed: int):
    if seed:
        np.random.seed(seed)
    
    flattened_matrix = loc.t_values.flatten()
    flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)
    n_units = int(flattened_matrix.size * percentage)
    filtered_matrix = flattened_matrix[flattened_matrix != -np.inf] # Filtered -inf value

    # Generate a random mask for the filtered matrix
    random_indices = np.random.choice(filtered_matrix.size, n_units, replace=False)
    mask_filtered = np.zeros_like(filtered_matrix, dtype=bool)
    mask_filtered[random_indices] = True

    # Map the mask back to the original flattened matrix
    mask_random = np.zeros_like(flattened_matrix, dtype=bool)
    valid_indices = np.where(flattened_matrix != -np.inf)[0]
    mask_random[valid_indices] = mask_filtered

    # Reshape the mask to the original matrix shape if needed
    mask_random = mask_random.reshape(loc.t_values.shape)

    return mask_random


def get_masked_both_tails(loc: LocImportantUnits, percentage: float):
    if percentage==0:
        return np.zeros_like(loc.t_values, dtype=np.int64)
    abs_tvalues = np.abs(loc.t_values)
    num_top_elements = int(abs_tvalues.size * percentage)

    # Flatten the matrix, find the threshold value for the top 1%
    flattened_matrix = abs_tvalues.flatten()

    # Replace NaN with a sentinel value (e.g., -inf)
    flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)

    # Filter out -inf values
    filtered_matrix = flattened_matrix[flattened_matrix != -np.inf]
    if len(filtered_matrix) > num_top_elements:
        threshold_value = np.partition(flattened_matrix, -num_top_elements)[-num_top_elements]
    else:
        threshold_value = -np.inf  # or handle this case as needed      

    # Create a binary mask where 1 represents the top 1% elements, and 0 otherwise
    mask_units = np.where(abs_tvalues >= threshold_value, 1, 0)
    return mask_units