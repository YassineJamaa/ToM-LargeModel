class ImportLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # # Ensure the tokenizer has a padding token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_embd_size(self):
        return self.model.config.hidden_size

    def get_nb_layers(self):
        return self.model.config.num_hidden_layers