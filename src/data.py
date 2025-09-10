import numpy as np
from medmnist import BloodMNIST

def _to_numpy(ds):
    imgs = np.array(ds.imgs)
    y = np.array(ds.labels).ravel().astype(np.int64)
    if imgs.ndim == 4 and imgs.shape[-1] == 3:
        imgs = imgs.mean(axis=-1) # this is based on grayscale
    return imgs, y

def _flatten(x):
    return x.reshape(x.shape[0], -1).astype(np.float32)

def _standardize(x_ref, x):
    mu = x_ref.mean(axis=0, keepdims=True)
    sd = x_ref.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd

def load_data(size=28, subsample_train=None, seed=0):
    # DO THE SPLITS
    tr = BloodMNIST(split="train", download=True, size=size)
    va = BloodMNIST(split="val", download=True, size=size)
    te = BloodMNIST(split="test", download=True, size=size)

    Xtr_img, ytr = _to_numpy(tr)
    Xva_img, yva = _to_numpy(va)
    Xte_img, yte = _to_numpy(te)

    # Flatten to vectors
    Xtr = _flatten(Xtr_img)
    Xva = _flatten(Xva_img)
    Xte = _flatten(Xte_img)
