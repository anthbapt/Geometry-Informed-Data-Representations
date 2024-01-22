import networkx as nx
import collections
import numpy as np
from functools import partial
import operator

def _calculate_new_faces(faces, new, old_set):
    """
    Calculates the new triangular faces for the network when we 
    add in new

    Parameters
    -----------
    faces : list
        a list of the faces present in the graph
    new : int
        the node id that is being added to the face
    old_set : set
        the old face that the node is being added to

    Returns
    -------
    None
    """
    faces.remove(frozenset(old_set))

    faces.add(frozenset([new, old_set[1], old_set[0]]))
    faces.add(frozenset([new, old_set[0], old_set[2]]))
    faces.add(frozenset([new, old_set[1], old_set[2]]))

    
def _add_triangular_face(G, new, old_set, C, faces):
    """
    Adds a new triangle to the networkx graph G

    Parameters
    -----------
    G : networkx graph
        the networkx graph to add the new face to
    new : int
        the node id that is being added to the face
    C : array_like
        correlation matrix
    old_set : set
        the old face that the node is being added to
    faces : list
        a list of the faces present in the graph

    Returns
    -------
    None
    """
    if isinstance(new, collections.Sized):
        raise ValueError("New should be a scaler")

    if len(old_set) > 3:
        raise ValueError("Old set is not the right size!")
    for j in old_set:
        G.add_edge(new, j, weight=C[new, j])

        
def tmfg(corr, absolute=False, threshold_mean=True):
    """
    Constructs a TMFG from the supplied correlation matrix

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix
    absolute : bool
        whether to use the absolute correlation values for chooisng weights or normal ones
    threshold_mean : bool
        this will discard all correlations below the mean value when selecting the first 4 
        vertices, as in the original implementation

    Returns
    -------
    networkx graph
        The Triangular Maximally Filtered Graph
    """
    p = corr.shape[0]

    if absolute:
        weight_corr = np.abs(corr)
    else:
        weight_corr = corr

    # Find the 4 most central vertices
    new_weight = weight_corr.copy()
    if threshold_mean:
        new_weight[new_weight < new_weight.mean()] = 0
    degree_centrality = new_weight.sum(axis=0)
    ind = np.argsort(degree_centrality)[::-1]
    starters = ind[0:4]
    starters_set = set(starters)
    not_in = set(range(p))
    not_in = not_in.difference(starters_set)
    G = nx.Graph()
    G.add_nodes_from(range(p))

    # Add the tetrahedron in
    faces = set()
    _add_triangular_face(G, ind[0], set([ind[1], ind[3]]), corr, faces)
    _add_triangular_face(G, ind[1], set([ind[2], ind[3]]), corr, faces)
    _add_triangular_face(G, ind[0], set([ind[2], ind[3]]), corr, faces)
    _add_triangular_face(G, ind[2], set([ind[1], ind[3]]), corr, faces)

    faces.add(frozenset([ind[0], ind[1], ind[3]]))
    faces.add(frozenset( [ind[1], ind[2], ind[3]] ))
    faces.add(frozenset([ind[0], ind[2], ind[3]]))
    faces.add(frozenset([ind[0], ind[1], ind[2]]))

    while len(not_in) > 0:
        #to_check = permutations(starters_set, 3)

        max_corr = -np.inf
        max_i = -1
        nodes_correlated_with = None
        not_in_arr = np.array(list(not_in))

        # Find the node most correlated with the faces in the TMFG currently
        for ind in faces:
            ind = list(ind)
            ind_arr = np.array(ind)
            most_related = weight_corr[ind_arr, :][:, not_in_arr].sum(axis=0)
            ind_2 = np.argsort(most_related)[::-1]
            curr_corr = most_related[ind_2[0]]

            if curr_corr > max_corr:
                max_corr = curr_corr
                max_i = not_in_arr[[ind_2[0]]]
                nodes_correlated_with = ind

        starters_set = starters_set.union(set(max_i))
        not_in = not_in.difference(starters_set)
        _add_triangular_face(G, max_i[0], nodes_correlated_with, corr, faces)
        _calculate_new_faces(faces, max_i[0], nodes_correlated_with)
    
    return G
