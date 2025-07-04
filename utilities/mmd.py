import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_ST = SentenceTransformer('all-mpnet-base-v2')

def MMD(X, Y, kernel="rbf", device=None):
    """
    Compute the empirical Maximum Mean Discrepancy (MMD) between two samples X and Y.

    Args:
        X (torch.Tensor): Samples from distribution P, shape (n, m)
        Y (torch.Tensor): Samples from distribution Q, shape (l, m)
        kernel (str): Kernel type, "rbf" or "multiscale"
        device (torch.device or None): Device to run on (e.g., torch.device("cuda"))
    
    Returns:
        torch.Tensor: Scalar MMD distance between the two distributions.
    """
    if device is None:
        device = X.device

    X = X.to(device)
    Y = Y.to(device)

    n = X.size(0)
    l = Y.size(0)

    # Compute squared Euclidean distances
    XX_dist = torch.cdist(X, X, p=2).pow(2)  # (n, n)
    YY_dist = torch.cdist(Y, Y, p=2).pow(2)  # (l, l)
    XY_dist = torch.cdist(X, Y, p=2).pow(2)  # (n, l)

    # Initialize kernels
    if kernel == "rbf":
        bandwidths = [10, 15, 20, 50]
        def compute_kernel(dists):
            K = 0
            for a in bandwidths:
                K += torch.exp(-0.5 * dists / a)
            return K

    elif kernel == "multiscale":
        bandwidths = [0.2, 0.5, 0.9, 1.3]
        def compute_kernel(dists):
            K = 0
            for a in bandwidths:
                K += (a**2) / (a**2 + dists)
            return K

    else:
        raise ValueError("Unknown kernel. Use 'rbf' or 'multiscale'.")

    # Apply kernel
    K_xx = compute_kernel(XX_dist)
    K_yy = compute_kernel(YY_dist)
    K_xy = compute_kernel(XY_dist)

    # MMD estimate
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


