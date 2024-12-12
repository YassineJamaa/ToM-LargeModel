class ImportVLM:
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    
    def get_embd_size(self):
        return self.model.language_model.config.hidden_size

    def get_nb_layers(self):
        return self.model.language_model.config.num_hidden_layers