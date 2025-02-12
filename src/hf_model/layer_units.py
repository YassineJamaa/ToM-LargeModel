import torch
from torch.utils.data import Dataset
from .import_model import ImportModel
from benchmark import BenchmarkBaseline
from tqdm import tqdm
from functools import partial


def average_tokens_layers(activation):
    return activation.mean(dim=1)

def final_tokens_layers(activation):
    return activation[:, -1, :, :]

class LayerUnits:
    def __init__(self, 
                 import_model: ImportModel, 
                 localizer: Dataset,
                 device, 
                 method:str="average",
                 args_tokenizer: dict={}):
        self.import_model = import_model
        self.language_model = self.import_model.get_language_model()
        self.data = localizer
        self.device = device
        self.data_activation = None
        self.group = {"positive": 0, "negative": 1}

        # Dynamically choose the right inputs function
        if self.import_model.model_type == "LLM":
            self.get_inputs = partial(self.get_inputs_llm, **args_tokenizer)
        elif self.import_model.model_type == "VLM":
            self.get_inputs = self.get_inputs_vlm
        else:
            raise ValueError(f"Unsupported model type: {self.import_model.model_type}")
        
        self.tokens_agg = average_tokens_layers if method == "average" else final_tokens_layers if method == "final" else None
        self.name = type(localizer).__name__

        # clear hooks
        self.extract_all_units()
        self.clear_hooks()
    
    def reset_data_activation(self):
        embd_size = self.import_model.get_embd_size()
        n_layers = self.import_model.get_layers()
        self.data_activation = torch.zeros(2, len(self.data), embd_size, n_layers, device=self.device)
    
    def clear_hooks(self):
        for layer in self.language_model.model.layers:
            layer._forward_hooks.clear()
    
    def reset(self):
        self.clear_hooks()
        self.reset_data_activation()

    def get_hook_layers(self, idx, activation):
        def hook_layers(module, input, output):
            activation[:, :, :, idx] = output[0].squeeze(0).to(self.device)
        return hook_layers

    def get_inputs_llm(self, text, **kwargs):
        return self.import_model.tokenizer(text, return_tensors="pt", **kwargs).to(self.device)

    def get_inputs_vlm(self, text, **kwargs):
        return self.import_model.processor(text, return_tensors="pt", **kwargs).to(self.device)

    def extract_layer_units(self, idx, group_name="positive"):
        self.clear_hooks()
        self.import_model.model.eval()

        text = self.data[idx][self.group[group_name]]
        inputs = self.get_inputs(text)
        n_tokens = inputs["input_ids"].shape[1]
        embd_size = self.import_model.get_embd_size()
        n_layers = self.import_model.get_layers()
        activation = torch.zeros(1, n_tokens, embd_size, n_layers, device=self.device)

        for i, layer in enumerate(self.language_model.model.layers):
            layer.register_forward_hook(self.get_hook_layers(i, activation))

        with torch.no_grad():
            self.import_model.model(**inputs)
        return self.tokens_agg(activation)

    def extract_all_units(self):
        self.reset()
        for idx in range(len(self.data)):
            self.data_activation[0, idx, :, :] = self.extract_layer_units(idx, "positive")
            self.data_activation[1, idx, :, :] = self.extract_layer_units(idx, "negative")

class AverageTaskStimuli:
    def __init__(self,
                 benchmark: BenchmarkBaseline,
                 import_model: ImportModel,
                 device: str):
        self.benchmark = benchmark
        self.import_model = import_model
        self.data_activation = None
        self.language_model = self.import_model.get_language_model()
        self.device = device 

        # Dynamically choose the right inputs function
        if self.import_model.model_type == "LLM":
            self.get_inputs = self.get_inputs_llm
        elif self.import_model.model_type == "VLM":
            self.get_inputs = self.get_inputs_vlm
            self.get_row = self.get_row_llm
        else:
            raise ValueError(f"Unsupported model type: {self.import_model.model_type}")
        
        if type(self.benchmark).__bases__[0].__name__ == "BenchmarkText":
            self.get_row = self.get_row_llm
        elif type(self.benchmark).__bases__[0].__name__ == "BenchmarkVisionText":
            self.get_row = self.get_row_vlm
        else:
            raise ValueError(f"Unsupported Benchmark type: {self.import_model.model_type}")
        
        self.avg_activation = self.extract_all_units().to("cpu")
        self.clear_hooks()
    
    def reset_data_activation(self):
        embd_size = self.import_model.get_embd_size()
        n_layers = self.import_model.get_layers()
        self.data_activation = torch.zeros(len(self.benchmark), embd_size, n_layers, device=self.device)
    
    def clear_hooks(self):
        for layer in self.language_model.model.layers:
            layer._forward_hooks.clear()
    
    def reset(self):
        self.clear_hooks()
        self.reset_data_activation()
    
    def get_hook_layers(self, idx, activation):
        def hook_layers(module, input, output):
            activation[:, :, idx] = output[0].squeeze(0).to(self.device)
        return hook_layers
    
    def get_row_llm(self, idx):
        return f"{self.benchmark[idx]} {self.benchmark.data["answer"].iloc[idx]}"

    def get_row_vlm(self, idx):
        text, _ = self.benchmark[idx]
        return f"{text} Answer: {self.benchmark.data["answer"].iloc[idx]}"

    def get_inputs_llm(self, text):
        return self.import_model.tokenizer(text, return_tensors="pt").to(self.device)

    def get_inputs_vlm(self, text):
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.import_model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        return self.import_model.processor(text=prompt, return_tensors="pt").to(self.device)
    
    def extract_layer_units(self, idx):
        self.clear_hooks()
        self.import_model.model.eval()
        text = self.get_row(idx)  
        inputs = self.get_inputs(text)
        n_tokens = inputs["input_ids"].shape[1]
        embd_size = self.import_model.get_embd_size()
        n_layers = self.import_model.get_layers()

        activation = torch.zeros(n_tokens, embd_size, n_layers, device=self.device)
        for i, layer in enumerate(self.language_model.model.layers):
            layer.register_forward_hook(self.get_hook_layers(i, activation))
        
        with torch.no_grad():
            self.import_model.model(**inputs)
        
        return activation.mean(dim=0)
    
    def extract_all_units(self):
        self.reset()
        for idx in tqdm(range(len(self.benchmark)), desc="Processing Mean Imputation"):
            self.data_activation[idx,:,:] = self.extract_layer_units(idx)
        return self.data_activation.mean(dim=0)




        

    
    
    


    



    

    

