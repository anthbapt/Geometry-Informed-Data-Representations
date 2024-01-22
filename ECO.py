import numpy as np

def eco(W, directed = False):
    """
    Filter a weighted square similarity matrix to obtain an adjacency matrix.

    Parameters:
    - W: Weighted square similarity matrix.
    - directed: Boolean (1/0) indicating whether the network is directed or undirected.

    Returns:
    - A: Filtered adjacency matrix.

    Reference:
    "A topological criterion to filter information in complex brain networks, 
    De Vico Fallani et al, Plos Comp Biol, 2017"
    """

    N = W.shape[0]

    if directed:
        numcon = 3 * N
        ind = np.where(W != 0)
    else:
        W = np.triu(W)
        numcon = int(1.5 * N)
        ind = np.where(np.triu(W) != 0)

    if numcon > len(ind[0]):
        raise ValueError('Input matrix is too sparse')

    sorted_indices = ind[0][np.argsort(W[ind])[::-1][::]]
    W[sorted_indices[numcon:]] = 0

    if directed:
        A = np.double(np.logical_not(np.isclose(W, 0)))
    else:
        A = np.double(np.logical_not(np.isclose(W + W.T, 0)))

    return A
