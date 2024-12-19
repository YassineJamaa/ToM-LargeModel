def get_language_model(model):
    """
    Dynamically retrieves the language model from the provided model instance,
    adapting to whether it is wrapped in DataParallel or not.

    Args:
        model (torch.nn.Module): The model instance to retrieve the language model from.

    Returns:
        torch.nn.Module: The language model component.
    """
    if hasattr(model, "module"):  # Check if the model is wrapped in DataParallel
        return model.module.language_model
    elif hasattr(model, "language_model"):
        return model.language_model
    else:
        raise AttributeError("The provided model does not have a 'language_model' attribute.")
