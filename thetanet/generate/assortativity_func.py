import numpy as np
import thetanet as tn


def a_func_transform(A, k_in):
    """ Create an assortativity function based on the transformation of an
    adjacency matrix A.

    Combine all neurons with the same in-degree and average.
    Since some in-degrees may not occur in the particular realisation of this
    network represented through A, one follows the convention, that E still
    holds space for the whole in-degree space. As a consequence there will be
    some zeros in E.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.
    k_in : array_like, 1D int
        In-degree space.

    Returns
    -------
    E : array_like, 2D float
        Assortativity function of transformation-kind.
    B : array_like, 2D float
        Transformation matrix.
    """

    K_in = A.sum(1)
    N = len(K_in)
    N_k_in = len(k_in)

    # Lifting operator
    B = np.zeros((N, N_k_in))
    B[np.arange(N), K_in - k_in.min()] = 1
    occurring_degrees = (B.sum(0) != 0)

    # Reduction operator
    C = np.copy(B).T
    C[occurring_degrees] /= C[occurring_degrees].sum(1)[:, None]

    E = np.matmul(C, np.matmul(A, B))

    return E, B


def a_func_empirical(A, k_in, P_k_in, k_out, P_k_out, i_prop='in',
                     j_prop='out'):
    """ Create an assortativity function based on counting connections of an
    adjacency matrix.

    Parameters
    ----------
    A : array_like, 2D int
        Adjacency matrix.
    k_in : array_like, 1D int
        In-degree space.
    P_k_in : array_like, 1D float
        In-degree probability.
    k_out : array_like, 1D int
        Out-degree space.
    P_k_out : array_like, 1D float
        Out-degree probability.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    a : array_like, 2D float
        Empirical assortativity function.
    """

    K_in = A.sum(1)
    K_out = A.sum(0)
    N = A.shape[0]

    K_i_prop = eval('K_' + i_prop)
    K_j_prop = eval('K_' + j_prop)
    k_i_prop = eval('k_' + i_prop)
    k_j_prop = eval('k_' + j_prop)
    P_k_i_prop = eval('P_k_' + i_prop)
    P_k_j_prop = eval('P_k_' + j_prop)

    edges = np.argwhere(A > 0)
    i = edges[:, 0]
    j = edges[:, 1]
    degree_combis = np.asarray([K_j_prop[j]-min(k_j_prop),
                                K_i_prop[i]-min(k_i_prop)])
    unique_degree_combis, counts = np.unique(degree_combis, axis=1,
                                             return_counts=True)
    count_func = np.zeros((len(k_j_prop), len(k_i_prop)))
    count_func[unique_degree_combis[0], unique_degree_combis[1]] += counts

    a = count_func / N**2 / P_k_j_prop[:, None] / P_k_i_prop[None, :]

    return a


def r_from_a_func(a, k_out, k_in, k_mean):
    """ Compute assortativity coefficient r from a assortativity function a.

    Parameters
    ----------
    a : array_like, 2D float
        Assortativity function.
    k_in : array_like, 1D int
        In-degree space.
    k_out : array_like, 1D int
        Out-degree space.

    Returns
    -------
    r : float
        Assortativity coefficient.
    """

    mesh_k_out, mesh_k_in = np.meshgrid(k_out, k_in)
    cor = np.sum((mesh_k_out-k_mean) * (mesh_k_in-k_mean) * a)
    std_out = np.sqrt(np.sum((mesh_k_out-k_mean)**2 * a))
    std_in = np.sqrt(np.sum((mesh_k_in-k_mean)**2 * a))
    r = cor / std_out / std_in

    return r


