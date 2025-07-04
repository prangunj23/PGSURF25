import numpy as np
import ot

def wasserstein(X, Y):
    
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]

    M = ot.dist(X, Y)
    M /= M.max()

    return ot.emd2(a, b, M)