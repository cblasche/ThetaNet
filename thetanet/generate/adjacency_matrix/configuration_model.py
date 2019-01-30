import numpy as np
import thetanet as tn
from time import time


def configuration_model(K_in, K_out, r=0, i_prop='in', j_prop='out'):
    """ Compute an adjacency matrix from in- and out-degree sequences using the
    configuration model.

    Parameters
    ----------
    K_in : array_like, 1D int
        In-degree sequence.
    K_out : array_like, 1D int
        Out-degree sequence.
    r : float
        Assortativity coefficient.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    A : array_like, 2D int
        Adjacency matrix.
    """

    print('Adjacency matrix - Configuration model | N =', len(K_in), '| K_in = [',
          min(K_in), ',', max(K_in), '] | K_out = [', min(K_out), ',',
          max(K_out), '] | r =', r)

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

    if r != 0:
        print('\r    Assortative mixing:')
        assortative_mixing(A, r, i_prop, j_prop)
        print('      Successful.')

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
    num_multi_edges = np.sum(A) - num_unique_edges
    edges = np.append(unique_edges, np.zeros((num_multi_edges, 2), dtype=int), axis=0)
    multi_edges = np.argwhere(A > 1)

    # loop through multi-edges
    for I in multi_edges:
        while A[I[0], I[1]] > 1:
            picked = np.random.choice(num_unique_edges)
            J = np.copy(edges[picked])
            if I[0] != J[1] and J[0] != I[1]:  # avoid self-edges
                if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:   # avoid more
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


def assortative_mixing(A, r, i_prop='in', j_prop='out'):
    """ Mix connections in adjacency matrix to achieve desired assortativity of
    respective properties of pre-(j) and post-(i)-synaptic neurons.
    First take a selection of the available edges, split this selection and
    built pairs. Check for each pair if reconnecting is a good decision.
    Second check if assortativity coefficient is not already too extreme. If
    so, reduce the number of edge pairs.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.
    r : float
        Assortativity coefficient.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    A will be modified.
    """

    def swap_criteria(I, J, K_i_prop, K_j_prop, K_mean, r_diff):
        """ Compute for 2 lists of edges I and J if they should be swapped.
        Criteria are based on changing the assortativity in the desired
        direction as well as to avoid further self-edges and minimize
        multi-edges.

        Parameters
        ----------
        I : array_like, 1D int
            An edge from A. I[0] is post-synaptic and I[1] is pre-synaptic.
        J : array-like, 1D int
            An edge from A. J[0] is post-synaptic and J[1] is pre-synaptic.
        K_i_prop : array_like, 1D int
            Degree sequence of post-synaptic neurons (i) of respective property.
        K_j_prop : array_like, 1D int
            Degree sequence of pre-synaptic neurons (j) of respective property.
        K_mean : float
            Mean degree of degree sequence.
        r_diff : float
            Difference between current and target assortativity coefficient.

        Returns
        -------
        swap : array_like, 1D bool
            Boolean value if pair should be swapped.
        """

        # Criteria: adjust assortativity in desired direction
        cor_original = (K_i_prop[I[0]] - K_mean) * (K_j_prop[I[1]] - K_mean) + \
                       (K_i_prop[J[0]] - K_mean) * (K_j_prop[J[1]] - K_mean)
        cor_swap = (K_i_prop[I[0]] - K_mean) * (K_j_prop[J[1]] - K_mean) + \
                   (K_i_prop[J[0]] - K_mean) * (K_j_prop[I[1]] - K_mean)
        if r_diff > 0:
            criteria_cor = (cor_swap > cor_original)
        else:
            criteria_cor = (cor_swap < cor_original)

        # Criteria: avoid multi-edges directly when adding 1 to certain entries
        criteria_multi = np.logical_and((A[J[0], I[1]] == 0), (A[I[0], J[1]] == 0))

        # Criteria: avoid self-edges when adding 1 to the diagonal
        criteria_self = np.logical_and((I[0] != J[1]), (J[0] != I[1]))

        # Criteria: avoid multi-edges when adding 1 when it has been done
        # already by another pair of edges
        # Note: This is not dynamic, hence there will be some multi-edges.
        first_occurence_JI = np.unique([J[0], I[1]], axis=1,
                                       return_index=True)[1]
        first_occurence_IJ = np.unique([I[0], J[1]], axis=1,
                                       return_index=True)[1]
        criteria_unique = np.full((2, I.shape[1]), False)
        criteria_unique[0][first_occurence_IJ] = True
        criteria_unique[1][first_occurence_JI] = True
        criteria_unique = np.logical_and(criteria_unique[0], criteria_unique[1])

        # apply all criteria
        swap = (criteria_cor * criteria_multi * criteria_self *
                criteria_unique).astype(bool)

        return swap

    # Compute sequences
    K_in = A.sum(axis=1)
    K_out = A.sum(axis=0)
    K_mean = np.sum(K_in ** 2) / A.sum()    # K_in_mean and K_out_mean are equal

    # Match them to their respective property of pre-(j) and post-(i)-synaptic
    # neuron
    K_i_prop = eval('K_' + i_prop)
    K_j_prop = eval('K_' + j_prop)

    # Converge to desired assortativity
    r_diff = r - assortativity_coefficient(A, i_prop, j_prop)
    iteration = 0
    limited_edges = int(A.sum())    # start with considering all edges
    while abs(r_diff) > 0.01:
        iteration += 1
        # Determine pairs of edges to swap
        edges = np.argwhere(A)
        np.random.shuffle(edges)
        edges = edges[:limited_edges]  # limit the speed of approaching r
        if len(edges) % 2 == 0:
            I, J = np.split(edges.T, 2, axis=1)
        else:
            I, J = np.split(edges[:-1].T, 2, axis=1)
        swap = swap_criteria(I, J, K_i_prop, K_j_prop, K_mean, r_diff)
        if swap.any() != True:
            print('Desired assortativity coefficient r=', r, 'is impossible ',
                  'to reach for this network configuration.')
            return
        # New assortativity coefficient can be too extreme and needs to be
        # checked with a dummy therefore.
        A_dummy = np.copy(A)
        reconnect_edge_pair(A_dummy, I[:, swap], J[:, swap])
        remove_multi_edges(A_dummy)
        r_diff_dummy = r - assortativity_coefficient(A_dummy, i_prop, j_prop)
        if abs(r_diff_dummy) < abs(r_diff):
            A = A_dummy
            r_diff = r_diff_dummy
            print('\r      Iteration', iteration, ': r_diff =', r_diff)
        else:
            limited_edges = int(limited_edges * 0.3)
            print('\r      Iteration', iteration, ': Step size has been reduced.')

    return


def assortativity_coefficient(A, i_prop='in', j_prop='out'):
    """ Compute the assortativity coefficient with respect to chosen
    properties. Connections are from neuron j to i.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    r : float
        Assortativity coefficient.
    """

    K_in = A.sum(axis=1)
    K_out = A.sum(axis=0)
    K_i_prop = eval('K_' + i_prop)
    K_j_prop = eval('K_' + j_prop)

    N = len(K_in)
    K_mean = K_in.mean()

    j, i = np.meshgrid(range(N), range(N))  # order j, i is correct
    var_i_prop = np.sum(A * (K_i_prop[i] - K_mean) ** 2)
    var_j_prop = np.sum(A * (K_j_prop[j] - K_mean) ** 2)
    cor = np.sum(A * (K_i_prop[i] - K_mean) * (K_j_prop[j] - K_mean))
    r = cor / (np.sqrt(var_i_prop) * np.sqrt(var_j_prop))

    return r
