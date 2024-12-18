import torch

def setup_device():
    """
    Detects system resources and selects the appropriate device for computation.
    Returns:
        - device: The selected device (CPU or CUDA)
        - device_ids: List of GPUs used (if applicable)
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✅ CUDA is available! Number of GPUs detected: {num_gpus}")
        
        # List all GPUs
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set primary GPU and return device list
        device = torch.device("cuda:0")
        print(f"👉 Using GPU: {torch.cuda.get_device_name(0)}")
        device_ids = list(range(num_gpus))  # [0, 1, ..., num_gpus-1]
    else:
        print("❌ CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
        device_ids = []
    
    return device, device_ids