import os
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def get_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    print("Device:", get_device())
    print("Torch:", torch.__version__)