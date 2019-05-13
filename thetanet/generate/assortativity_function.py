import numpy as np
from scipy import ndimage
from scipy import interpolate
import thetanet as tn

"""
For assortativity functions we follow the convention of adjacency matrices:
First two axes contain node degrees of receiving neurons and the second axis
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


def a_func_empirical2d(A, k_in, P_k_in, k_out, P_k_out, i_prop, j_prop,
                       smooth_function=True):
    """ Create an assortativity function based on counting connections of an
    adjacency matrix.
    (Use degree spaces with no gaps.)

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
    smooth_function : bool
        Apply gaussian filter to smooth the finite size noise.

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

    if smooth_function:
        a = ndimage.gaussian_filter(a, [len(k_in)/20, len(k_out)/20])

    return a


def r_from_a_func(a, k_in, k_out, P_k, i_prop, j_prop):
    """ Compute the assortativity coefficient from a 4D assortativity function.
    Project 4D function to the relevant 2D one and compute r from it.

    Parameters
    ----------
    a : ndarray, 4D float
        Assortativity function.
    k_in : ndarray, 1D int
        In-degree space.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k : ndarray, 2D float
        Joint In-/Out-degree probability.
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

    P_k_in = P_k.sum(1)
    P_k_out = P_k.sum(0)
    k_mean = P_k_in.dot(k_in)

    k_mean_edge_i_in = P_k_in.dot(k_in ** 2) / P_k_in.dot(k_in)
    k_mean_edge_i_out = np.tensordot(P_k, np.outer(k_in, k_out))/k_mean
    k_mean_edge_j_in = k_mean_edge_i_out
    k_mean_edge_j_out = P_k_out.dot(k_out ** 2) / P_k_out.dot(k_out)

    k_i = eval('k_' + i_prop)
    k_j = eval('k_' + j_prop)
    P_k_i = eval('P_k_' + i_prop)
    P_k_j = eval('P_k_' + j_prop)
    k_mean_edge_i = eval('k_mean_edge_i_' + i_prop)
    k_mean_edge_j = eval('k_mean_edge_j_' + j_prop)

    if len(a.shape) != 2:
        a = a_func_4d_to_2d(a, P_k_in, P_k_out, i_prop, j_prop)

    count_func = a * np.outer(P_k_i, P_k_j)
    mesh_k_i, mesh_k_j = np.meshgrid(k_i, k_j, indexing='ij')
    cor = np.sum((mesh_k_i - k_mean_edge_i) * (mesh_k_j - k_mean_edge_j)
                 * count_func)
    std_i = np.sqrt(np.sum((mesh_k_i - k_mean_edge_i) ** 2 * count_func))
    std_j = np.sqrt(np.sum((mesh_k_j - k_mean_edge_j) ** 2 * count_func))
    r = cor / std_i / std_j

    return r


def a_func_4d_to_2d(a, P_k_in, P_k_out, i_prop, j_prop):
    """ Sum over irrelevant axes and leave those of i_prop and j_prop.
    This will reduce a 4d assortativity function to a 2d one.

    Parameters
    ----------
    a : ndarray, 4D float
        Assortativity function.
    P_k_in : ndarray, 1D float
        In-degree probability.
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
    a : float, 2D float
        Assortativity function.
    """

    # choose axis which is to be contracted
    i_mean_axis = {'in': 1, 'out': 0}
    j_mean_axis = {'in': 3, 'out': 2}
    # choose property of contraction
    mean_prop = {'in': 'out', 'out': 'in'}
    P_k_i_mean = eval('P_k_' + mean_prop[i_prop])
    P_k_j_mean = eval('P_k_' + mean_prop[j_prop])

    a = np.tensordot(a, P_k_j_mean, (j_mean_axis[j_prop], 0))
    a = np.tensordot(a, P_k_i_mean, (i_mean_axis[i_prop], 0))

    return a


def a_func_linear(k_in, k_out, P_k, N, c, i_prop, j_prop):
    """ Assortativity function.
    Based on the linear approach for neutral assortativity and extended with
    the addition method to introduce assortativity.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    k_out : ndarray, 1D float
        Out-degree probability.
    P_k : ndarray, 2D float
        Joint In-/Out-degree distribution.
    N : int
        Number of neurons.
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
    a : ndarray, 4D float
        Assortativity function.
    """

    P_k_in = P_k.sum(1)
    P_k_out = P_k.sum(0)
    k_mean = P_k_in.dot(k_in)

    k_mean_edge_i_in = P_k_in.dot(k_in ** 2) / P_k_in.dot(k_in)
    k_mean_edge_i_out = np.tensordot(P_k, np.outer(k_in, k_out))/k_mean
    k_mean_edge_j_in = k_mean_edge_i_out
    k_mean_edge_j_out = P_k_out.dot(k_out ** 2) / P_k_out.dot(k_out)

    if i_prop == 'in':
        i_prop_axis = 0
    elif i_prop == 'out':
        i_prop_axis = 1
    if j_prop == 'in':
        j_prop_axis = 2
    elif j_prop == 'out':
        j_prop_axis = 3
    i_dim_array = np.ones(4, int)
    i_dim_array[i_prop_axis] = -1
    j_dim_array = np.ones(4, int)
    j_dim_array[j_prop_axis] = -1

    k_i = eval('k_' + i_prop).reshape(i_dim_array)
    k_j = eval('k_' + j_prop).reshape(j_dim_array)
    k_mean_edge_i = eval('k_mean_edge_i_' + i_prop)
    k_mean_edge_j = eval('k_mean_edge_j_' + j_prop)

    a = k_in[:, None, None, None] * \
        np.ones(len(k_out))[None, :, None, None] * \
        np.ones(len(k_in))[None, None, :, None] * \
        k_out[None, None, None, :]
    a += c * (k_j - k_mean_edge_j) * (k_i - k_mean_edge_i)
    a /= N * k_mean
    a = np.clip(a, 0, 1)

    return a


def a_func_linear_r(k_in, k_out, P_k, N, i_prop, j_prop):
    """ Compute the valid range of assortativity parameter c to get the whole
     range of assortativity coefficients r which are possible to reach with
     for this network.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k : ndarray, 2D float
        Joint In-/Out-degree probability.
    N : int
        Number of neurons.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    r_arr : ndarray, 1D float
        Array of assortativity coefficients.
    c_arr : ndarray, 1D float
        Array of corresponding assortativity parameters.
    """

    def r_from_c(c):
        a = a_func_linear(k_in, k_out, P_k, N, c, i_prop, j_prop)
        return r_from_a_func(a, k_in, k_out, P_k, i_prop, j_prop)

    c_arr = [0., 1.]
    r_arr = [r_from_c(c) for c in c_arr]
    while r_arr[-1] - r_arr[-2] > 1e-4:
        c_arr.append(0.05 * len(c_arr) / (r_arr[1] - r_arr[0]))
        r_arr.append(r_from_c(c_arr[-1]))

    c_arr = np.asarray(c_arr)
    c_arr_neg = -1 * c_arr[1:][::-1]
    r_arr_neg = [r_from_c(c) for c in c_arr_neg]

    c_arr = np.append(c_arr_neg, c_arr)
    r_arr = np.append(r_arr_neg, r_arr)
    r_arr = np.asarray(r_arr)

    def a_func(r):
        try:
            c = interpolate.interp1d(r_arr, c_arr, kind='cubic')(r)
        except ValueError:
            print('\nCorrelation r =', r, 'is out of the interpolation'
                  ' range. Try smaller absolute values.')
            quit(1)
        a = a_func_linear(k_in, k_out, P_k, N, c, i_prop, j_prop)
        return a

    return a_func
