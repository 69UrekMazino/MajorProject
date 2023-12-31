import torch

# Check if CUDA (NVIDIA GPU library) is available
if torch.cuda.is_available():
    print("CUDA is available! GPU detected.")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")  # Print the GPU device name
else:
    print("CUDA is not available. Running on CPU.")
