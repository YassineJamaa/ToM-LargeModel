from src.ablation.zeroing import ZeroingAblation
import numpy as np

class MeanImputation(ZeroingAblation):
    def __init__(self, mean_units: np.ndarray):
        super().__init__()
        self.mean_units = mean_units
    
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
            unit_values = self.mean_units[unit_indices, idx].to(output[0].device)
            output[0][:, :, unit_indices] = unit_values.to(dtype=output[0].dtype)
        return hook_ablate

