import sys

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("TF GPU available:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TensorFlow error:", e)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch GPU available:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch error:", e)

try:
    import mne
    print("MNE version:", mne.__version__)
except Exception as e:
    print("MNE error:", e)

try:
    import pymatreader
    print("pymatreader installed successfully")
except Exception as e:
    print("pymatreader error:", e)

try:
    import sklearn
    print("scikit-learn version:", sklearn.__version__)
except Exception as e:
    print("scikit-learn error:", e)

print("Python version:", sys.version)
