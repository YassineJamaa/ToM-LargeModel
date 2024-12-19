
def get_model_component(model):
    """
    Dynamically retrieves the model component from the provided model instance,
    adapting to whether it is wrapped in DataParallel or not.

    Args:
        model (torch.nn.Module): The model instance to retrieve the language model from.

    Returns:
        torch.nn.Module: The language model component.
    """
    if hasattr(model, "module"):
        return model.module
    return model


class ImportModel:
    def __init__(self, model_type, model, tokenizer, processor=None):
        self.model_type = model_type
        self.model = model
        self.model_ref = get_model_component(self.model)
        self.config = self.get_config()
        self.tokenizer = tokenizer
        self.processor = processor
    
    def get_language_model(self):
        if self.model_type == "LLM":
            return self.model_ref
        elif self.model_type == "VLM":
            return self.model_ref.language_model
        
    def get_config(self):
        if self.model_type == "LLM":
            return self.model_ref.config
        elif self.model_type == "VLM":
            return self.model_ref.language_model.config
    
    def get_embd_size(self):
        return self.config.hidden_size
        
    def get_layers(self):
        return self.config.num_hidden_layers
        