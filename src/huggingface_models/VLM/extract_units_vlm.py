import torch
from torch.utils.data import Dataset
from .import_vlm import ImportVLM

class LayersUnitsVLM:
    def __init__(self, vlm: ImportVLM, localizer_data: Dataset):
        self.vlm = vlm
        self.data = localizer_data
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_activation = None
        self.group = {"positive": 0, "negative": 1}
        self.method_fn = "average"
        self.name = type(localizer_data).__name__

        self.extract_all_units()
        self.clear_hooks()

    def reset_data_activation(self):
        embd_size = self.vlm.get_embd_size()
        n_layers = self.vlm.get_nb_layers()
        # Move the tensor to GPU
        self.data_activation = torch.zeros(2, len(self.data), embd_size, n_layers, device=self.device)

    def clear_hooks(self):
            for layer in self.vlm.model.language_model.model.layers:
                layer._forward_hooks.clear()
    
    def reset(self):
        self.clear_hooks()
        self.reset_data_activation()

    def get_hook_layers(self, idx, activation):
        def hook_layers(module, input, output):
            activation[:, :, :, idx] = output[0].squeeze(0).to(self.device)
        return hook_layers
    
    def average_tokens_layers(self, activation):
        return activation.mean(dim=1)
    
    def extract_layer_units(self, idx, group_name="positive"):
        self.clear_hooks()
        self.vlm.model.eval()

        text = self.data[idx][self.group[group_name]]
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.vlm.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.vlm.processor(text=prompt, return_tensors='pt').to(self.device)
        
        n_tokens = inputs["input_ids"].shape[1]
        embd_size = self.vlm.get_embd_size()
        n_layers = self.vlm.get_nb_layers()
        
        activation = torch.zeros(1, n_tokens, embd_size, n_layers, device=self.device)

        for i, layer in enumerate(self.vlm.model.language_model.model.layers):
            layer.register_forward_hook(self.get_hook_layers(i, activation))

        with torch.no_grad():
            self.vlm.model(**inputs)

        return self.average_tokens_layers(activation)

    def extract_all_units(self):
        self.reset()
        for idx in range(len(self.data)):
            self.data_activation[0, idx, :, :] = self.extract_layer_units(idx, "positive")
            self.data_activation[1, idx, :, :] = self.extract_layer_units(idx, "negative")