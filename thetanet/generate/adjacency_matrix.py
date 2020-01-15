import numpy as np
import thetanet as tn
from time import time
from scipy import interpolate


def configuration_model(K_in, K_out, r=0, i_prop='in', j_prop='out',
                        simple=True, console_output=True):
    """ Compute an adjacency matrix from in- and out-degree sequences using the
    configuration model.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    r : float
        Assortativity coefficient.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    simple: bool
        Remove self- and multi-edges if True to create simple network. If False
        leave the those edges in the network.
    console_output: bool
        Whether or not to print details to the console.

    Returns
    -------
    A : ndarray, 2D int
        Adjacency matrix.
    """

    if console_output:
        print('\nCreating adjacency matrix: Configuration model')
        print('|......................................')
        print('| N =', K_in.shape[0], 'and N_edges =', K_in.sum())
        print('| k_in = [', min(K_in), ',', max(K_in), ']')
        print('| k_out = [', min(K_out), ',', max(K_out), ']')
        if r != 0:
            print('| r(', j_prop, '-', i_prop, ') =', r)
        print('|......................................')
        print('\r| Edge list:', end=' ')

    edges = edges_from_sequence(K_in, K_out)

    if console_output:
        print('Successful.')
        print('\r| Adjacency matrix:', end=' ')

    A = matrix_from_edges(edges)

    if console_output:
        print('Successful.')
    if simple:
        if console_output:
            print('\r| Remove self-edges:', end=' ')

        remove_self_edges(A)

        if console_output:
            print('Successful.')
            print('\r| Remove multi-edges:', end=' ')

            remove_multi_edges(A, console_output=False)

            print('Successful.')

    if r != 0:
        if console_output:
            print('\r| Assortative mixing:', end=' ')
        if simple:
            assortative_mixing(A, r, i_prop, j_prop, console_output=False)
        else:
            assortative_mixing(A, r, i_prop, j_prop, console_output=False,
                               eliminate_multi_edges=False)
        if console_output:
            print('Successful.')

    return A


def edges_from_sequence(K_in, K_out):
    """ Generate a list containing the edges of the network.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.

    Returns
    -------
    edges : ndarray, 2D int
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
    edges : ndarray, 2D int
        Directed edges from edges[:, 1] to edges[:, 0].

    Returns
    -------
    A : ndarray, 2D int
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
    A : ndarray, 2D int
        Adjacency matrix.
    I : ndarray, 1D int
        An edge from A. I[0] is post-synaptic and I[1] is pre-synaptic.
    J : ndarray, 1D int
        An edge from A. J[0] is post-synaptic and J[1] is pre-synaptic.

    Returns
    -------
    A will be modified.
    """
    if len(I.shape) > 1:
        if A[A>1].sum() > 0:
            I0I1, I0I1_counts = np.unique(np.asarray([I[0], I[1]]), axis=1,
                                          return_counts=True)
            J0J1, J0J1_counts = np.unique(np.asarray([J[0], J[1]]), axis=1,
                                          return_counts=True)
            I0J1, I0J1_counts = np.unique(np.asarray([I[0], J[1]]), axis=1,
                                          return_counts=True)
            J0I1, J0I1_counts = np.unique(np.asarray([J[0], I[1]]), axis=1,
                                          return_counts=True)
            A[I0I1[0], I0I1[1]] -= I0I1_counts
            A[J0J1[0], J0J1[1]] -= J0J1_counts
            A[I0J1[0], I0J1[1]] += I0J1_counts
            A[J0I1[0], J0I1[1]] += J0I1_counts
        else:
            I0J1, I0J1_counts = np.unique(np.asarray([I[0], J[1]]), axis=1,
                                          return_counts=True)
            J0I1, J0I1_counts = np.unique(np.asarray([J[0], I[1]]), axis=1,
                                          return_counts=True)
            A[I[0], I[1]] -= 1
            A[J[0], J[1]] -= 1
            A[I0J1[0], I0J1[1]] += I0J1_counts
            A[J0I1[0], J0I1[1]] += J0I1_counts
    else:
        A[I[0], I[1]] -= 1
        A[J[0], J[1]] -= 1
        A[I[0], J[1]] += 1
        A[J[0], I[1]] += 1

    return


def remove_self_edges(A):
    """ If adjacency matrix has non-zero entries on its diagonal (self-edges)
    they will be removed by swapping connections. This will not affect degree
    distribution and degree correlation.

    Parameters
    ----------
    A : ndarray, 2D int
        Adjacency matrix.

    Returns
    -------
    A will be modified.
    """

    edges = np.empty((0, 2), dtype=int)
    for i in range(1, A.max() + 1):
        edges = np.append(edges, np.argwhere(A >= i), axis=0)
    num_edges = edges.shape[0]
    self_edge_indices = np.flatnonzero(edges[:, 0] == edges[:, 1])

    for self_edge_index in self_edge_indices:
        I = np.copy(edges[self_edge_index])
        A_I_init = np.copy(A[I[0], I[1]])
        while A[I[0], I[1]] == A_I_init:
            rand_index = np.random.choice(num_edges)
            J = np.copy(edges[rand_index])
            # avoid multi-edges
            if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:
                reconnect_edge_pair(A, I, J)
                # update edges
                edges[rand_index] = [J[0], I[1]]
                edges[self_edge_index] = [I[0], J[1]]

    return


def remove_multi_edges(A, console_output=False):
    """  If adjacency matrix has entries higher than 1 (multi-edge) they will
    be removed by swapping connections. This will not affect degree distribution
    and node correlation.

    Parameters
    ----------
    A : ndarray, 2D int
        Adjacency matrix.
    console_output: bool
        Whether or not to print details to the console.

    Returns
    -------
    A will be modified.
    """

    if console_output:
        print('\nRemoving multi-edges')
        print('|......................................')
        print('| N =', A.shape[0], 'and N_edges =', A.sum())
        print('| N_multi_edges =', A[A > 1].sum())
        print('|......................................')

    runtime_start = time()

    edges = np.empty((0, 2), dtype=int)
    for i in range(1, A.max()+1):
        edges = np.append(edges, np.argwhere(A >= i), axis=0)
    num_edges = edges.shape[0]
    multi_indices = np.argwhere(A[edges[:, 0], edges[:, 1]] > 1).squeeze()

    for multi_index in multi_indices:
        I = np.copy(edges[multi_index])
        A_I_init = np.copy(A[I[0], I[1]])
        if A_I_init > 1:
            while A[I[0], I[1]] == A_I_init:
                rand_index = np.random.choice(num_edges)
                J = np.copy(edges[rand_index])
                # avoid self-edges
                if I[0] != J[1] and J[0] != I[1]:
                    # avoid additional multi-edges
                    if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:
                        reconnect_edge_pair(A, I, J)
                        # update edges
                        edges[multi_index] = [I[0], J[1]]
                        edges[rand_index] = [J[0], I[1]]

    if console_output:
        print('| N_multi_edges =', A[A > 1].sum())
        print('|......................................')
        print('| runtime:', np.round(time()-runtime_start, 1), 'sec\n')

    return


def assortative_mixing(A, r, i_prop='in', j_prop='out', console_output=True,
                       eliminate_multi_edges=True, precision=0.001):
    """ Mix connections in adjacency matrix to achieve desired assortativity of
    respective properties of pre-(j) and post-(i)-synaptic neurons.
    Firstly, take a selection of the available edges, split this selection and
    built pairs. Check for each pair if reconnecting is a good decision.
    Secondly, check if assortativity coefficient is not already exceeded. If
    so, reduce the number of edge pairs of that last iteration and repeat.
    Estimate the reduction based on a interpolation approach.

    Parameters
    ----------
    A : ndarray, 2D int
        Adjacency matrix.
    r : float
        Assortativity coefficient.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    console_output: bool
        Whether or not to print details to the console.
    eliminate_multi_edges: bool
         Remove multi_edges within the scheme.
    precision: float (positive)
        Define how close the algorithm has to approach r.

    Returns
    -------
    A will be modified.
    """

    def selection_criteria(A, I, J, r_diff):
        # Criteria 1: reconnect edges only if it benefits the edge correlation
        K_in = A.sum(axis=1)
        K_out = A.sum(axis=0)
        num_edges = K_in.sum()
        K_i_prop = eval('K_' + i_prop)
        K_j_prop = eval('K_' + j_prop)
        K_mean_edge_i = (K_i_prop[np.append(I[0], J[0])]).sum() / num_edges
        K_mean_edge_j = (K_i_prop[np.append(I[1], J[1])]).sum() / num_edges
        cor_original = (K_i_prop[I[0]] - K_mean_edge_i) * \
                       (K_j_prop[I[1]] - K_mean_edge_j) + \
                       (K_i_prop[J[0]] - K_mean_edge_i) * \
                       (K_j_prop[J[1]] - K_mean_edge_j)
        cor_swapped = (K_i_prop[I[0]] - K_mean_edge_i) * \
                      (K_j_prop[J[1]] - K_mean_edge_j) + \
                      (K_i_prop[J[0]] - K_mean_edge_i) * \
                      (K_j_prop[I[1]] - K_mean_edge_j)
        if r_diff > 0:
            criteria_cor = (cor_swapped > cor_original)
        else:
            criteria_cor = (cor_swapped < cor_original)

        # Criteria 2: avoid multi-edges directly when adding 1 to certain entries
        criteria_multi = np.logical_and((A[J[0], I[1]] == 0),
                                        (A[I[0], J[1]] == 0))

        # Criteria 3: avoid self-edges when adding 1 to the diagonal
        criteria_self = np.logical_and((I[0] != J[1]), (J[0] != I[1]))

        select = (criteria_cor * criteria_multi * criteria_self).astype(bool)

        if not select.any():
            print('Desired assortativity coefficient r=', r, 'is impossible ',
                  'to reach for this network configuration.')
            return

        return select

    def reconnect_selection(A, I, J, select):
        """
        I and J the edges of A split in two lists of equal size. Reconnect
        reconnect pairs of edges (I,J) only when they are selected according to
        the 'select array.
        """
        A_trial = np.copy(A)
        reconnect_edge_pair(A_trial, I[:, select], J[:, select])
        if eliminate_multi_edges:
            tn.generate.remove_multi_edges(A_trial)
        r_diff_trial = r - tn.generate.r_from_matrix(A_trial, i_prop, j_prop)
        return A_trial, r_diff_trial

    def edges(A):
        """
        Return a shuffled list of edges in A.
        Dimension 0: in-out-node
        Dimension 1: as long as there are edges
        """
        edges = np.empty((0, 2), dtype=int)
        for i in range(1, A.max() + 1):
            edges = np.append(edges, np.argwhere(A >= i), axis=0)
        num_edges = edges.shape[0]
        rand_indices = np.random.choice(num_edges, num_edges, replace=False)
        return edges[rand_indices].T

    def refine_selection(A, I, J, select, r_diff, r_diff_trial):
        """
        Adjust r by deselecting an appropriate amount of the edges for the
        reconnection process.
        The amount of edges to deselect is interpolated on the basis of known
        data points.
        """
        if console_output:
            print('\r|        refining last iteration:')

        r_diff_interp = [r_diff, r_diff_trial]
        limit_interp = [0, I.shape[1]]
        interp_kind = 'linear'
        limit = int(interpolate.interp1d(r_diff_interp, limit_interp,
                                         kind=interp_kind)(0))

        iteration_refine = 0
        while abs(r_diff_trial) > precision:
            iteration_refine += 1
            select_trial = np.copy(select)
            select_trial[limit:] = False

            A_trial, r_diff_trial = reconnect_selection(A, I, J, select_trial)

            # Add every point to the pool to increase interpolation accuracy.
            r_diff_interp.append(r_diff_trial)
            limit_interp.append(limit)
            # After collecting enough data, change to cubic interpolation
            if len(r_diff_interp) >= 4:
                interp_kind = 'cubic'
            limit = int(interpolate.interp1d(r_diff_interp, limit_interp,
                                             kind=interp_kind)(0))
            if console_output:
                print('\r|        ', iteration, '_', iteration_refine, ': r =',
                      np.round(r - r_diff_trial, 5))
        return A_trial, r_diff_trial

    runtime_start = time()
    r_diff = r - tn.generate.r_from_matrix(A, i_prop, j_prop)
    if console_output:
        print('\nAssortative mixing of type (', j_prop, ',', i_prop, ')')
        print('|......................................')
        print('| N =', A.shape[0], 'and N_e =', A.sum())
        if eliminate_multi_edges:
            print('| removing multi-edges: enabled')
        else:
            print('| removing multi-edges: disabled')
        print('| current r =', np.round(r - r_diff, 3))
        print('| target r =', np.round(r, 3))
        print('| precision:', precision)
        print('|......................................')
    iteration = 0

    while abs(r_diff) > precision:
        iteration += 1
        if (A.sum() % 2) == 0:
            I, J = np.split(edges(A), 2, axis=1)
        else:
            I, J = np.split(edges(A)[:, :-1], 2, axis=1)
        select = selection_criteria(A, I, J, r_diff)
        A_trial, r_diff_trial = reconnect_selection(A, I, J, select)

        if console_output:
            print('\r|    ', iteration, ': r =',
                  np.round(r - r_diff_trial, 5))

        # Before accepting this process, check if we already went beyond the
        # target r. If so, the sign in r_diff changes.
        # In this case no further iteration is required, since we can achieve
        # the desired r by reducing the amount of selected edge pairs.
        if abs(r_diff_trial) > precision \
                and np.sign(r_diff_trial) != np.sign(r_diff):
            A_trial, r_diff_trial = refine_selection(A, I, J, select, r_diff,
                                                     r_diff_trial)

        # Copy values to A in-place
        A[...] = A_trial[...]
        r_diff = r_diff_trial

    if console_output:
        print('|......................................')
        print('| runtime:', np.round(time() - runtime_start, 1), 'sec')

    return


def chung_lu_model(K_in, K_out, k_in=None, k_out=None, P_k=None, c=0,
                   i_prop='in', j_prop='out'):
    """ Chung Lu style adjacency matrix creation. Generating a target matrix T
    and compare it with a matrix filled with random variables. Assortativity
    may be induced by a non-zero c value.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    k_in : ndarray, 1D int
        In-degree space.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k : ndarray, 1D float
        Joint In-Out-degree probability.
    c : float
        Assortativity parameter.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    A : ndarray, 2D int
        Adjacency matrix.
    """

    N = K_in.shape[0]

    T = np.outer(K_in, K_out) / (N * K_in.mean())

    if c != 0:
        if np.any([array is None for array in [k_in, k_out, P_k]]):
            print('Chung Lu model requires k_in, k_out and P_k in order to compute'
                  ' correct mean values.')
            quit()
        P_k_in = P_k.sum(1)
        P_k_out = P_k.sum(0)
        k_mean = P_k_in.dot(k_in)

        k_mean_edge_i_in = P_k_in.dot(k_in ** 2) / k_mean
        k_mean_edge_i_out = (P_k * np.outer(k_in, k_out)).sum() / k_mean
        k_mean_edge_j_in = k_mean_edge_i_out
        k_mean_edge_j_out = P_k_out.dot(k_out ** 2) / k_mean

        k_mean_edge_i = eval('k_mean_edge_i_' + i_prop)
        k_mean_edge_j = eval('k_mean_edge_j_' + j_prop)
        K_i = eval('K_' + i_prop)
        K_j = eval('K_' + j_prop)

        T += c * np.outer((K_i - k_mean_edge_i), (K_j - k_mean_edge_j)) \
            / (N * K_in.mean())

    A = np.random.uniform(size=(N, N)) < T

    return A.astype(int)


def r_from_matrix(A, i_prop='in', j_prop='out'):
    """ Compute the assortativity coefficient with respect to chosen
    properties. Connections are from neuron j to i.

    Parameters
    ----------
    A : ndarray, 2D int
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

    N = A.shape[0]

    K_in = A.sum(axis=1)
    K_out = A.sum(axis=0)

    edges = np.argwhere(A > 0)
    i, j = edges[:, 0], edges[:, 1]

    K_i_prop = eval('K_' + i_prop)
    K_j_prop = eval('K_' + j_prop)
    K_mean_edge_i = (A[i, j]*K_i_prop[i]).sum() / A.sum()
    K_mean_edge_j = (A[i, j]*K_j_prop[j]).sum() / A.sum()

    i, j = np.meshgrid(range(N), range(N), indexing='ij')
    var_i_prop = np.sum(A * (K_i_prop[i] - K_mean_edge_i) ** 2)
    var_j_prop = np.sum(A * (K_j_prop[j] - K_mean_edge_j) ** 2)
    cor = np.sum(A * (K_i_prop[i] - K_mean_edge_i) *
                 (K_j_prop[j] - K_mean_edge_j))
    r = cor / (np.sqrt(var_i_prop) * np.sqrt(var_j_prop))

    return r


def rho_from_matrix(A):
    """ Compute Pearson correlation coefficient for an adjacency matrix.

    Parameters
    ----------
    A : ndarray, 2D int
        Adjacency matrix.

    Returns
    -------
    rho : float
        Pearson correlation coefficient.
    """

    K_in = A.sum(1)
    K_out = A.sum(0)

    rho = tn.generate.rho_from_sequences(K_in, K_out)

    return rho


