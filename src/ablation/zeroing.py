from src.huggingface_models.LLM import ImportLLM
import numpy as np

class ZeroingAblation:
    def __init__(self):
        pass

    def clear_hooks(self, model_wrapper):
        """
        Clears all registered forward hooks in the model layers.

        Args:
            model_wrapper: An instance of ImportLLM or ImportVLM.
        """
        if hasattr(model_wrapper.model, 'model'):  # For ImportLLM
            layers = model_wrapper.model.model.layers
            print("Model type detected: LLM")
        elif hasattr(model_wrapper.model.language_model, 'model'):  # For ImportVLM
            layers = model_wrapper.model.language_model.model.layers
            print("Model type detected: VLM")
        else:
            raise ValueError("Unsupported model wrapper type. Ensure the structure matches either ImportLLM or ImportVLM.")

        for layer in layers:
            layer._forward_hooks.clear()

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
            output[0][:, :, unit_indices] = 0
        return hook_ablate
    
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
            output[0][:, :, unit_indices] = self.mean_units[:, unit_indices]
        return hook_ablate