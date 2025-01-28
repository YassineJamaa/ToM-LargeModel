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
    def __init__(self, model_type, model, tokenizer, processor=None, chat_template=None):
        self.model_type = model_type # Inform whether the model is VLM or LLM
        self.model = model # Set the model

        # Dynamically retrieves the model components
        self.model_ref = get_model_component(self.model)
        self.config = self.get_config()

        # Set the tokenizer & processor 
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding = False
        self.processor = processor
        self.method_process = None
        
        # Add Chat Template to Tokenizer & Processor
        #self.tokenizer.chat_template = chat_template
        if self.processor:
            if chat_template:
                self.processor.chat_template = chat_template
            self.method_process = self.processor
        else:
            self.method_process = self.tokenizer

    
    def get_language_model(self):
        """ Dynamically retrieves the language model for VLM and LLM """
        if self.model_type == "LLM":
            return self.model_ref
        elif self.model_type == "VLM":
            return self.model_ref.language_model
        
    def get_config(self):
        """ Dynamically retrieves the model's config """
        if self.model_type == "LLM":
            return self.model_ref.config
        elif self.model_type == "VLM":
            return self.model_ref.language_model.config
    
    def get_embd_size(self):
        """ Return the embedding vector size """
        return self.config.hidden_size
        
    def get_layers(self):
        """ Return the number of Transformer Blocks """
        return self.config.num_hidden_layers
        