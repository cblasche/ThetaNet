import numpy as np


def configuration_model(K_in, K_out):
    """ Compute an adjacency matrix from in- and out-degree sequences using the
    configuration model.

    Parameters
    ----------
    K_in : array_like, 1D int
        In-degree sequence.
    K_out : array_like, 1D int
        Out-degree sequence.

    Returns
    -------
    A : array_like, 2D int
        Adjacency matrix.
    """

    print('Adjacency matrix - Configuration model | N =', len(K_in), '| K_in = [',
          min(K_in), ',', max(K_in), '] | K_out = [', min(K_out), ',',
          max(K_out), '] | Generating:')

    print('\r    Edge list:', end=' ')
    edges = edges_from_sequence(K_in, K_out)
    print('Successful.')

    print('\r    Adjacency matrix:', end=' ')
    A = matrix_from_edges(edges)
    print('Successful.')

    print('\r    Remove self-edges:', end=' ')
    remove_self_edges(A)
    print('Successful.')

    print('\r    Remove multi-edges:', end=' ')
    remove_multi_edges(A)
    print('Successful.')

    return A


def edges_from_sequence(K_in, K_out):
    """ Generate a list containing the edges of the network.

    Parameters
    ----------
    K_in : array_like, 1D int
        In-degree sequence.
    K_out : array_like, 1D int
        Out-degree sequence.

    Returns
    -------
    edges : array_like, 2D int
        Edges of the shape: edges[:, 0] post-synaptic neurons
                            edges[:, 1] pre-synaptic neurons
    """

    edges = np.empty((K_in.sum(), 2), dtype=int)    # list of edges holding
                                                    # neurons labels
    i_in = 0
    i_out = 0
    for i in range(len(K_in)):
        edges[i_in:i_in + K_in[i], 0] = [i] * K_in[i]
        edges[i_out:i_out + K_out[i], 1] = [i] * K_out[i]
        i_in += K_in[i]
        i_out += K_out[i]
    np.random.shuffle(edges[:, 1])  # randomise who is connected to whom

    return edges


def matrix_from_edges(edges):
    """ Build an adjacency matrix A from edge list, where
    A[i, j] != 0 if neuron j's output is connected to neuron i's input or
            = 0 if not.
    Edges may be repeated in 'edges', which leads to A[i,j] > 1.

    Parameters
    ----------
    edges : array_like, 2D int
        Directed edges from edges[:, 1] to edges[:, 0].

    Returns
    -------
    A : array-like, 2D int
        Adjacency matrix.
    """

    N = edges.max()+1
    A = np.zeros((N, N), dtype=int)  # adjacency matrix A
    unique_edges, counts = np.unique(edges, return_counts=True, axis=0)
    A[(unique_edges[:, 0], unique_edges[:, 1])] = counts

    return A


def reconnect_edge_pair(A, I, J):
    """ Reconnect a pair of edges (I and J) in the adjacency matrix A. I is an
    edge with a connection from I[1] to I[0] and the same holds for J.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.
    I : array_like, 1D int
        An edge from A. I[0] is post-synaptic and I[1] is pre-synaptic.
    J : array-like, 1D int
        An edge from A. J[0] is post-synaptic and J[1] is pre-synaptic.

    Returns
    -------
    A will be modified.
    """

    A[I[0], I[1]] -= 1
    A[J[0], J[1]] -= 1
    A[J[0], I[1]] += 1
    A[I[0], J[1]] += 1

    return


def remove_self_edges(A):
    """ If adjacency matrix has non-zero entries on its diagonal (self-edges)
    they will be removed by swapping connections. This will not affect degree
    distribution and degree correlation.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.

    Returns
    -------
    A will be modified.
    """

    edges = np.argwhere(A > 0)
    unique_edges = np.unique(edges, axis=0)
    num_unique_edges = len(unique_edges)
    overflow = np.sum(A) - num_unique_edges
    edges = np.append(unique_edges, np.zeros((overflow, 2), dtype=int), axis=0)
    self_edges_indices = np.flatnonzero(unique_edges[:, 0] ==
                                        unique_edges[:, 1])     # index of self-
                                                                # edges in edges

    # loop through self-edges
    for self_edges_index in self_edges_indices:
        I = np.copy(edges[self_edges_index])
        while A[I[0], I[1]] > 0:
            picked = np.random.choice(num_unique_edges)
            J = np.copy(edges[picked])
            if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:  # avoid multi-edges
                reconnect_edge_pair(A, I, J)

                # update edges
                if A[J[0], J[1]] == 0:
                    edges[picked] = [J[0], I[1]]
                else:  # if picked edge is a multi-edge create new unique edge
                    num_unique_edges += 1
                    edges[num_unique_edges - 1] = [J[0], I[1]]
                if A[I[0], I[1]] == 0:
                    edges[self_edges_index] = [I[0], J[1]]
                else:  # if self-edge is also multi-edge create new unique edge
                    num_unique_edges += 1
                    edges[num_unique_edges - 1] = [I[0], J[1]]

    return


def remove_multi_edges(A):
    """  If adjacency matrix has entries higher than 1 (multi-edge) they will
    be removed by swapping connections. This will not affect degree distribution
    and degree correlation.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.

    Returns
    -------
    A will be modified.
    """

    edges = np.argwhere(A > 0)
    unique_edges = np.unique(edges, axis=0)
    num_unique_edges = len(unique_edges)
    overflow = np.sum(A) - num_unique_edges
    edges = np.append(unique_edges, np.zeros((overflow, 2), dtype=int), axis=0)
    multi_edges = np.argwhere(A > 1)

    # loop through multi-edges
    for I in multi_edges:
        while A[I[0], I[1]] > 1:
            picked = np.random.choice(num_unique_edges)
            J = np.copy(edges[picked])
            if I[0] != J[1] and J[0] != I[1]:  # avoid self-edges
                if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:   # avoid
                                                                # multi-edges
                    reconnect_edge_pair(A, I, J)

                    # update edges
                    num_unique_edges += 1
                    edges[num_unique_edges - 1] = [I[0], J[1]]
                    if A[J[0], J[1]] == 0:
                        edges[picked] = [J[0], I[1]]
                    else:   # if picked edge is a multi-edge create new unique
                            # edge
                        num_unique_edges += 1
                        edges[num_unique_edges - 1] = [J[0], I[1]]

    return
