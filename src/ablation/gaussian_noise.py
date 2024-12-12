from src.ablation.zeroing import ZeroingAblation
import numpy as np
import torch

class GaussianNoise(ZeroingAblation):
    def __init__(self, mean_units: np.ndarray, std_err: float):
        super().__init__()
        self.mean_units = mean_units
        self.std_err = std_err
        self.noise = torch.normal(mean=0, std=self.std_err, size=self.mean_units.shape, device=self.mean_units.device) 
    
    def get_hook_ablate(self, idx, mask):
        """
        Defines a hook function to ablate specific units based on a mask.

        Args:
            idx (int): Layer index.
            mask (torch.Tensor): Binary mask for ablation at the given layer.

        Returns:
            function: A hook function to zero out specified units.
        """
        def hook_ablate(module, input, output):
            mask_layer = mask[idx]
            unit_indices = mask_layer.nonzero()[0]
            output[0][:, :, unit_indices] = self.mean_units[unit_indices, idx] + self.noise[unit_indices, idx]
        return hook_ablate