import sys
import torch
import torchaudio
import librosa
import numpy
import scipy
import sklearn
import tensorboard

print(f"Python      : {sys.version}")
print(f"PyTorch     : {torch.__version__}")
print(f"Torchaudio  : {torchaudio.__version__}")
print(f"Librosa     : {librosa.__version__}")
print(f"NumPy       : {numpy.__version__}")
print(f"SciPy       : {scipy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"TensorBoard : {tensorboard.__version__}")
print()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Device: CPU")

t = torch.ones(3, 3).to(device)
print(f"Tensor on {device}: {t.shape} — OK")
