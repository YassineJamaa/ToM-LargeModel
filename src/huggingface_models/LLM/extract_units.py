import torch
from torch.utils.data import Dataset
from .import_model import ImportLLM

class LayersUnitsLLM:
    def __init__(self, llm: ImportLLM, data: Dataset, method: str = "average"):
        self.llm = llm
        self.data = data
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_activation = None
        self.group = {"positive": 0, "negative": 1}
        self.method_fn = self.average_tokens_layers if method == "average" else self.final_tokens_layers if method == "final" else None
        self.name = type(data).__name__

        # Set tokenizer arguments based on dataset name
        self.tokenizer_args = {"return_tensors": "pt"}
        if self.name == "LangLocDataset":
            self.tokenizer_args.update({"truncation": True, "max_length": 12})

        self.extract_all_units()
        self.clear_hooks()

    def reset_data_activation(self):
        embd_size = self.llm.get_embd_size()
        n_layers = self.llm.get_nb_layers()
        # Move the tensor to GPU
        self.data_activation = torch.zeros(2, len(self.data), embd_size, n_layers, device=self.device)

    def clear_hooks(self):
        for layer in self.llm.model.model.layers:
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

    def final_tokens_layers(self, activation):
        return activation[:, -1, :, :]

    def extract_layer_units(self, idx, group_name="positive", method="average"):
        self.clear_hooks()
        self.llm.model.eval()

        prompt = self.data[idx][self.group[group_name]]
        inputs = self.llm.tokenizer(prompt, **self.tokenizer_args).to(self.device)
        
        n_tokens = inputs["input_ids"].shape[1]
        embd_size = self.llm.get_embd_size()
        n_layers = self.llm.get_nb_layers()
        
        activation = torch.zeros(1, n_tokens, embd_size, n_layers, device=self.device)

        for i, layer in enumerate(self.llm.model.model.layers):
            layer.register_forward_hook(self.get_hook_layers(i, activation))

        with torch.no_grad():
            self.llm.model(**inputs)

        return self.method_fn(activation)

    def extract_all_units(self):
        self.reset()
        for idx in range(len(self.data)):
            self.data_activation[0, idx, :, :] = self.extract_layer_units(idx, "positive")
            self.data_activation[1, idx, :, :] = self.extract_layer_units(idx, "negative")