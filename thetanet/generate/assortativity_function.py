import numpy as np
from scipy import optimize
import thetanet as tn

"""
For assortativity functions we follow the convention of adjacency matrices:
First axis contains node degrees of receiving neurons and the second axis
holds node degrees of sending neurons.
"""

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
    A : ndarray, 2D int
        Adjacency matrix.
    k_in : ndarray, 1D int
        In-degree space.

    Returns
    -------
    E : ndarray, 2D float
        Assortativity function of transformation-kind.
    B : ndarray, 2D float
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
    A : ndarray, 2D int
        Adjacency matrix.
    k_in : ndarray, 1D int
        In-degree space.
    P_k_in : ndarray, 1D float
        In-degree probability.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k_out : ndarray, 1D float
        Out-degree probability.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    a : ndarray, 2D float
        Empirical assortativity function.
    """

    K_in = A.sum(1)
    K_out = A.sum(0)
    N = A.shape[0]

    K_i = eval('K_' + i_prop)
    K_j = eval('K_' + j_prop)
    k_i = eval('k_' + i_prop)
    k_j = eval('k_' + j_prop)
    P_k_i = eval('P_k_' + i_prop)
    P_k_j = eval('P_k_' + j_prop)

    edges = np.argwhere(A > 0)
    i = edges[:, 0]
    j = edges[:, 1]
    degree_combis = np.asarray([K_i[i] - min(k_i),
                                K_j[j] - min(k_j)])
    unique_degree_combis, counts = np.unique(degree_combis, axis=1,
                                             return_counts=True)
    count_func = np.zeros((len(k_i), len(k_j)))
    count_func[unique_degree_combis[0], unique_degree_combis[1]] += counts

    a = count_func / N ** 2 / P_k_i[:, None] / P_k_j[None, :]

    return a


def a_coef_from_a_func(a, k_in, P_k_in, k_out, P_k_out, N, k_mean, i_prop='in',
                       j_prop='out'):
    """ Compute the assortativity coefficient from an assortativity function.
    Reconstruct 'count_func' one would gain from the adjacency matrix first and
    then compute correlation.

    Parameters
    ----------
    a : ndarray, 2D float
        Assortativity function.
    k_in : ndarray, 1D int
        In-degree space.
    P_k_in : ndarray, 1D float
        In-degree probability.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k_out : ndarray, 1D float
        Out-degree probability.
    N : int
        Number of neurons.
    k_mean : float
        Mean degree.
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

    k_i = eval('k_' + i_prop)
    k_j = eval('k_' + j_prop)
    P_k_i = eval('P_k_' + i_prop)
    P_k_j = eval('P_k_' + j_prop)

    count_func = a * np.outer(P_k_i, P_k_j) * N ** 2
    mesh_k_i, mesh_k_j = np.meshgrid(k_i, k_j)
    cor = np.sum((mesh_k_i - k_mean) * (mesh_k_j - k_mean) * count_func)
    std_j = np.sqrt(np.sum((mesh_k_i - k_mean) ** 2 * count_func))
    std_i = np.sqrt(np.sum((mesh_k_j - k_mean) ** 2 * count_func))
    r = cor / std_i / std_j

    return r


def a_func_linear(k_in, P_k_in, N, k_mean, r, rho):
    """ in/in-degree assortativity function.
    Based on the linear approach for neutral assortativity and extended with
    the exponent method to introduce assortativity. Since r cannot be included
    directly one has to numerically find the parameter to tune assortativity.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    P_k_in : ndarray, 1D float
        In-degree probability.
    N : int
        Number of neurons.
    k_mean : float
        Mean degree.
    r : float
        Assortativity coefficient.
    rho : float
        in/out-degree correlation.

    Returns
    -------
    a : ndarray, 2D float
        Assortativity function.
    """

    def a(c):
        exponent = (1 + np.outer(k_in - k_mean, k_in - k_mean) / k_mean ** 2) ** (-c)
        a = np.outer(k_in / N, (k_in / k_mean) ** rho) ** exponent
        return a

    def r_dif(c):
        r_d = a_coef_from_a_func(a(c), k_in, P_k_in, np.empty(0), np.empty(0),
                                 N, k_mean, i_prop='in', j_prop='in')
        r_d -= r
        return r_d

    c = optimize.root(r_dif, 0, method='hybr', tol=1e-2).x

    return a(c)


def rc_range(a, a_args):
    """ Compute the valid range of assortativity parameter c to get the whole
     range of assortativity coefficients r which are possible to reach with
     for this network.

    Parameters
    ----------
    a : function
        Assortativity function. First argument needs to be c, followed by the
        rest.
    a_args : tuple
        Function arguments of a, but without leading c.

    Returns
    -------
    r_arr : ndarray, 1D float
        Array of assortativity coefficients.
    c_arr : ndarray, 1D float
        Array of corresponding assortativity parameters.
    """

    k_in, P_k_in, N, k_mean, rho = a_args

    def r(c):
        return tn.generate.a_coef_from_a_func(a(c, *a_args), k_in, P_k_in, np.empty(0),
                                              np.empty(0), N, k_mean, i_prop='in',
                                              j_prop='in')

    c_arr = [0., 1.]
    r_arr = [r(c) for c in c_arr]

    while r_arr[-1] - r_arr[-2] > 1e-4:
        c_arr.append(0.05 * len(c_arr) / (r_arr[1] - r_arr[0]))
        r_arr.append(r(c_arr[-1]))

    c_arr = np.asarray(c_arr)
    c_arr_neg = -1 * c_arr[1:][::-1]
    r_arr_neg = [r(c) for c in c_arr_neg]

    c_arr = np.append(c_arr_neg, c_arr)
    r_arr = np.append(r_arr_neg, r_arr)
    r_arr = np.asarray(r_arr)

    return r_arr, c_arr


def a_param_from_a_coef(r, r_range, c_range):
    """ Return c value for a given r using the earlier calibrated r_range and
    c_range.

    Parameters
    ----------
    r : float
        Assortativity coefficient.
    r_range : ndarray, 1D float
        Array of assortativity coefficients.
    c_range : ndarray, 1D float
        Array of corresponding assortativity parameters.

    Returns
    -------
    c : float
        Assortativity parameter.
    """
    c = np.interp(r, r_range, c_range)
    return c