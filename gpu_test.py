import torch
import subprocess
import re

def get_driver_cuda_version():
    """
    Runs 'nvidia-smi' command to find the CUDA version reported by the driver.
    """
    try:
        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Use regex to find the CUDA Version string
        match = re.search(r'CUDA Version: (\d+\.\d+)', output)
        if match:
            return match.group(1)
        else:
            return "Not Found (nvidia-smi output changed?)"
            
    except FileNotFoundError:
        return "Not Found (nvidia-smi command not in PATH)"
    except Exception as e:
        return f"Error ({e})"

def check_gpu_setup():
    """
    Checks if PyTorch can detect and use the NVIDIA GPU and reports versions.
    """
    print(f"--- PyTorch GPU Check ---")
    
    # 1. Check PyTorch Version
    print(f"PyTorch Version: {torch.__version__}")

    # 2. Check if CUDA (GPU support) is available
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")

    if is_available:
        # 3. If available, get and print the GPU details
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        
        # --- NEW CHECKS ---
        pytorch_cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        driver_cuda_version = get_driver_cuda_version()
        
        print(f"\n--- Success! ---")
        print(f"Found {gpu_count} GPU(s).")
        print(f"Device 0 Name:   {gpu_name}")
        
        print(f"\n--- Version Report ---")
        print(f"Driver's CUDA Version:  {driver_cuda_version} (Max supported)")
        print(f"PyTorch CUDA Version:   {pytorch_cuda_version} (Compiled with)")
        print(f"cuDNN Version:          {cudnn_version}")

        print("\nThe 'realtime_3d.py' script WILL use your GPU.")
        print("Any lag is likely from the Matplotlib 3D graph, not the model.")
    
    else:
        # 4. If not available, print an error
        print(f"\n--- Warning ---")
        print("PyTorch cannot find your NVIDIA GPU (CUDA).")
        print("The 'realtime_3d.py' script will fall back to 'cpu'.")
        print("This is the main cause of the lag.")
        print("\nTo fix this, please ensure you have installed:")
        print("1. Your NVIDIA GTX 1650 drivers.")
        print("2. The correct version of PyTorch with CUDA support (e.g., pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118)")

if __name__ == "__main__":
    check_gpu_setup()

