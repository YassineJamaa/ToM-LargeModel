import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import ttest_ind
import numpy as np

class LocImportantUnits:
    def __init__(self,
                 checkpoint,
                 layers_units: torch.Tensor):
        self.model_name = checkpoint.split("/")[-1]
        self.fb_group = layers_units[0].cpu()
        self.fp_group = layers_units[1].cpu()
        self.t_values = self.welch_test()
        self.ranked_units = self.get_ranked_units()

    def welch_test(self):
        n_units = self.fb_group.shape[1]
        n_layers = self.fb_group.shape[2]

        # Reshape for Welch t-test
        fb_flattened = np.abs(self.fb_group.reshape(self.fb_group.shape[0], -1))
        fp_flattened = np.abs(self.fp_group.reshape(self.fp_group.shape[0], -1))
        
        # Perform the t-test along the first axis (sample dimension)
        t_stat, _ = ttest_ind(fb_flattened, fp_flattened, axis=0, equal_var=False)
        print(t_stat.shape)

        # Reshape t_stat back to (units, n_layers)
        return t_stat.reshape(n_units, n_layers)
    
    def get_ranked_units(self):
        # Get ranked matrix
        flat = self.t_values.flatten()
        sorted_indices = np.argsort(flat)[::-1]  # Sort indices in descending order
        ranked = np.empty_like(sorted_indices)
        ranked[sorted_indices] = np.arange(1, len(flat) + 1)
        # Reshape the ranked values back to the original matrix shape
        return ranked.reshape(self.t_values.shape)
    
    def get_masked_ktop(self, percentage):
        num_top_elements = int(self.t_values.size * percentage)

        # Flatten the matrix, find the threshold value for the top 1%
        flattened_matrix = self.t_values.flatten()

        # Replace NaN with a sentinel value (e.g., -inf)
        flattened_matrix = np.nan_to_num(flattened_matrix, nan=-np.inf)
        threshold_value = np.partition(flattened_matrix, -num_top_elements)[-num_top_elements]

        # Create a binary mask where 1 represents the top 1% elements, and 0 otherwise
        mask_units = np.where(self.t_values >= threshold_value, 1, 0)
        return mask_units
    
    def get_random_mask(self, percentage, seed=42):
        # Set the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate the total number of units
        total_units = self.t_values.size
        num_units_to_select = int(total_units * percentage)
        
        # Create a flattened array of zeros
        mask_flat = np.zeros(total_units, dtype=int)
        
        # Randomly select indices and set them to 1
        selected_indices = np.random.choice(total_units, num_units_to_select, replace=False)
        mask_flat[selected_indices] = 1
        
        # Reshape the mask back to the original shape
        return mask_flat.reshape(self.t_values.shape)
    
    def plot_layer_percentages(self, percentage, mask_type='ktop', seed=None, save_path=None):
        """
        Plots the percentage of important units per layer.
        
        Parameters:
        - percentage (float): The top percentage of units to be considered as important.
        - mask_type (str): Type of mask to use ('ktop' for k-top mask or 'random' for random mask).
        - seed (int, optional): Random seed for reproducibility when using the random mask.
        - save_path (str, optional): Path to save the plot. If None, shows the plot.
        """
        
        # Generate the mask based on the specified mask type
        if mask_type == 'ktop':
            mask = self.get_masked_ktop(percentage)
        elif mask_type == 'random':
            mask = self.get_random_mask(percentage, seed=seed)
        else:
            raise ValueError("Invalid mask_type. Choose 'ktop' or 'random'.")
        
        # Calculate the percentage of important units for each layer
        layer_percentages = [(np.sum(layer) / layer.size) * 100 for layer in mask.T]

        # Convert to a column vector for plotting
        layer_percentages_matrix = np.array(layer_percentages).reshape(-1, 1)

        # Plot the layer percentages as a matrix with shape (number of layers, 1)
        plt.figure(figsize=(2, 8))
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=min(layer_percentages), vmax=max(layer_percentages))
        plt.imshow(layer_percentages_matrix, cmap=cmap, aspect='auto')
        plt.colorbar(label="Percentage of Important Units")
        plt.title(f"Percentage of Important Units per Layer ({mask_type.capitalize()} Mask, Top {percentage*100:.1f}%)")
        plt.xlabel("Layer")
        plt.ylabel("Percentage")

        # Add text annotations with adaptive color based on background brightness
        for i, perc in enumerate(layer_percentages):
            color = cmap(norm(perc))
            brightness = 0.3 * color[0] + 0.5 * color[1] + 0.2 * color[2]
            text_color = "white" if brightness < 0.5 else "black"
            plt.text(0, i, f"{perc:.1f}%", ha="center", va="center", color=text_color)

        # Configure ticks
        plt.yticks(range(len(layer_percentages)), [f"Layer {i+1}" for i in range(len(layer_percentages))])
        plt.xticks([])

        # Save the plot or show it based on the save_path parameter
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        else:
            plt.show()