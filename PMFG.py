import scipy.sparse as sps
import networkx as nx
import numpy as np


def pmfg(W):
    """
    Construct a Planar Maximum Filtered Graph (PMFG) from a weighted, symmetric square matrix.

    Args:
        W (numpy.ndarray or scipy.sparse matrix): The input square matrix representing edge weights.
            The matrix must be real, symmetric, and square.

    Returns:
        scipy.sparse.lil_matrix: The PMFG as a sparse matrix in the LIL (List of Lists) format.

    Raises:
        ValueError: If the input matrix does not meet the required conditions.
        ValueError: If the PMFG cannot be found within a reasonable number of iterations.

    Note:
        The PMFG algorithm constructs a maximal planar graph from the input weighted matrix
        while preserving the planarity property. The resulting graph is represented as a sparse
        matrix in LIL format.

    References:
    """
    
    if W.shape[0] != W.shape[1] :
        print('The matrix must be square')
        sys.exit()
    if np.isreal(W.any()) != True :
        print('The matrix must be real')
        sys.exit()
    if W.any() != W.T.any() :
        print('The matrix must be symmetric')
        sys.exit()
    if sps.issparse(W.any()) == True :
        W = sps.lil_matrix(W)
        
    i, j, w = sps.find(sps.lil_matrix(W))
    kk = np.where(i < j)[0]
    ijw = np.column_stack((i[kk], j[kk], w[kk]))
    ind = np.argsort(-ijw[:,-1])
    ijw = ijw[ind]
    N = W.shape[0]
    P = sps.lil_matrix((N, N), dtype = W.dtype)
    
    for ii in range(min(6, ijw.shape[0])): # the first 6 edges from the list can be all inserted in a tetrahedron
        P[int(ijw[ii,0]), int(ijw[ii,1])] = ijw[ii,2]
        P[int(ijw[ii,1]), int(ijw[ii,0])] = ijw[ii,2]
    E = 6 # number of edges in P at this stage
    P1 = P.copy()
    ii = 6
    
    while E < 3*(N-2): # continue while all edges for a maximal planar graph are inserted
        ii += 1
        P1[int(ijw[ii,0]), int(ijw[ii,1])] = ijw[ii,2] # try to insert the next edge from the sorted list
        P1[int(ijw[ii,1]), int(ijw[ii,0])] = ijw[ii,2] # insert its reciprocal
        if nx.is_planar(nx.from_numpy_array(P1.todense())): # is the resulting graph planar?
            P = P1.copy() # Yes: insert the edge in P
            E += 1
        else:
            P1 = P.copy() # No: discard the edge
        if ii % 1000 == 0:
            #np.savez('P.npz', P=P, ii=ii)
            print(f'Build P: {ii}    :   {E/(3*(N-2))*100:.2f} per-cent done')
            if ii > N*(N-1)//2:
                print('PMFG not found')
                return P
    return P


#W = np.array([[1, 7, 1, 2, 2, 8, 7, 1, 1],
#              [7, 5, 9, 8, 5, 6, 0, 9, 2],
#              [1, 9, 1, 4, 1, 3, 4, 8, 9],
#              [2, 8, 4, 2, 2, 2, 4, 4, 6],
#              [2, 5, 1, 2, 3, 7, 4, 8, 9],
#              [8, 6, 3, 2, 7, 6, 2, 0, 0],
#              [7, 0, 4, 4, 4, 2, 7, 2, 4],
#              [1, 9, 8, 4, 8, 0, 2, 0, 8],
#              [1, 2, 9, 6, 9, 0, 4, 8, 1]])
#P = pmfg(W)