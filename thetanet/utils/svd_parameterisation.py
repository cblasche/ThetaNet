import numpy as np
import thetanet as tn


def svd_coef_E(A_stack, r_stack, pd_r, pd_k, k_in, m=2):
    """ Compute coefficients for the reconstruction of an assortativity function
    of transform-kind for any r within the range of r_stack.
    This is done starting from a stack of adjacency matrices with different
    assortativity coefficients. Potentially even an ensemble of realisation of
    each. The next step is generating transform-assortativity functions for
    each matrix in A_stack.
    After Singular Value Decomposition (SVD) is applied to those matrices
    the basis functions are approximated by polynomials. These polynomials as
    well as the singular values will be fitted with polynomials through
    assortativity space, i.e. the assortativity values.
    The coefficients of that fit are returned and serve as basis for later
    reconstruction.

    Parameters
    ----------
    A_stack : array_like, 3D/4D int
        Stack of adjacency matrices with different assortativity coefficients.
        Potentially several realisation of each (N_ensemble > 1)
        [(N_ensemble), r, N, N]
    r_stack : array_like, 1D float
        Stack of assortativity coefficients corresponding to A_stack.
    pd_r : int
        Maximum polynomial degree when fitting along assortativity space
        (r_stack).
    pd_k : int
        Maximum polynomial degree when fitting along degree space.
    k_in : array_like, 1D int
        In-degree space.
    m : int
        Order of reconstruction. After applying Singular Value Decomposition
        the first m functions and singular values will be processed for
        reconstruction.

    Returns
    -------
    coef : list
        List of coefficients of polynomials to reconstruct SVD-vectors and
        SVD-values.
    """

    u, v, s = svd_from_matrix(A_stack, r_stack, k_in, m)

    correct_signs(u)
    correct_signs(v)

    u_ens = average_ensemble(u)
    v_ens = average_ensemble(v)
    s_ens = average_ensemble(s)

    u_poly = polyfit_k(k_in, u_ens, pd_k)
    v_poly = polyfit_k(k_in, v_ens, pd_k)

    coef_u = coef_from_polyfit_r(r_stack, u_poly, pd_r)
    coef_v = coef_from_polyfit_r(r_stack, v_poly, pd_r)
    coef_s = coef_from_polyfit_r(r_stack, s_ens, pd_r)
    coef = [coef_u, coef_v, coef_s]

    return coef


def svd_E(coef, r):
    """ Reconstruct E using SVD. The corresponding SVD-values and vectors are
    reconstructed from polynomials, which's coefficients are stored in coef.

    Parameters
    ----------
    coef : list
        List of coefficients of polynomials to reconstruct svd-vectors and
        svd-values.
    r : float
        Assortativity coefficient which the reconstructed E-matrix shall have.

    Returns
    -------
    E : array_like, 2D float
        Assortativity function of transformation-kind with assortativity r.
    """

    coef_u, coef_v, coef_s = coef
    u = func_from_coef(coef_u, r)
    v = func_from_coef(coef_v, r)
    s = func_from_coef(coef_s, r)
    E = np.dot(np.dot(u.T, np.diag(s)), v)

    return E


def func_from_coef(coef, r):
    """ Polynomial reconstruction of functions for a given r.

    Parameters
    ----------
    coef : array_like, 2D/3D float
        Coefficients in ascending degree [Order, PolyDegree_r, (DegreeSpace)]
    r : float
        Assortativity coefficient.

    Returns
    -------
    f : array_like, 1D/2D
        Reconstructed basis function or singular value [Number, (DegreeSpace)]
    """

    if len(coef.shape) == 2:
        np.expand_dims(coef, axis=2)

    f = np.zeros((np.size(coef, 0), np.size(coef, 2)))
    for pd in range(np.size(coef, 1)):
        f += coef[:, pd, :] * r ** pd

    return f.squeeze()


def coef_from_polyfit_r(r_stack, y, pd_r):
    """ Fit for each degree of the degree space of basis vecotrs or singular
    values a polynomial in assortativity(r)-space and return coefficients.

    Parameters
    ----------
    r_stack : array_like, 1D float
        Stack of assortativity coefficients corresponding to A_stack.
    y : array_like, 2D/3D float
        Basis functions (recommended to use polynomial approximation)
        [Order, AssortativitySpace, (DegreeSpace)]
    pd_r : int
        Maximum polynomial degree when fitting along assortativity space
        (r_stack).

    Returns
    -------
    coef : array_like, 2D/3D float
        Coefficients in ascending degree [Order, PolyDegree_r, (DegreeSpace)]
    """

    if len(y.shape) == 2:
        np.expand_dims(y, axis=2)

    coef = np.zeros((np.size(y, 0), pd_r + 1, np.size(y, 2)))
    for m_i in range(np.size(y, 0)):
        coef[m_i] = np.polyfit(r_stack, y[m_i], pd_r)[::-1, :]  # coefficients come
        #  in opposite order

    return coef.squeeze()


def polyfit_k(k, y, pd_k):
    """ Polynomial approximation up to degree pd_k of basis functions y, where
     y is a stack of different order basis functions and a stack of different r.

    Parameters
    ----------
    k : array_like, 1D int
        Degree space.
    y : array_like, 3D float
        Basis functions to be approximated. [Order, AssortativitySpace, DegreeSpace]
    pd_k : int
        Maximum polynomial degree when fitting along degree space.

    Returns
    -------
    f : array_like, 3D float
        Polynomial approximation of basis functions.
        [Order, AssortativitySpace, (DegreeSpace)]
    """

    f = np.zeros((np.shape(y)))
    for m_i in range(np.size(f, 0)):
        for r_i in range(np.size(f, 1)):
            nonzero = y[m_i, r_i] != 0  # skip empty entries
            coef = np.polyfit(k[nonzero], y[m_i, r_i, nonzero], pd_k)
            f[m_i, r_i] = np.poly1d(coef)(k)

    return f


def average_ensemble(f):
    """ Average ensembles but let only nonzero entries contribute.

    Parameters
    ----------
    f : array_like, 3D/4D float
        SVD basis functions or singular values.
        [Ensemble, Order, AssortativitySpace, DegreeSpace]

    Returns
    -------
    sum : array_like, 2D/3D float
        Averaged basis function or singular values.
        [Order, AssortativitySpace, DegreeSpace]

    """

    contributing_ensembles = (f != 0).sum(0)
    sum = f.sum(0)
    sum[contributing_ensembles != 0] /= contributing_ensembles[contributing_ensembles != 0]

    return sum


def correct_signs(f):
    """ Flip signs of basis functions if necessary. Let mean values be positive
    (without loss of generality).

    Parameters
    ----------
    f : array_like, 3D/4D float
        SVD basis functions or singular values.
        [Ensemble, Order, AssortativitySpace, DegreeSpace]

    Returns
    -------
    f will be modified.
    """

    wrong_sign = f.mean(-1) < 0
    f[wrong_sign] = f[wrong_sign] * (-1)


def svd_from_matrix(A_stack, r_stack, k_in, m):
    """

    Parameters
    ----------
    A_stack : array_like, 3D/4D int
        Stack of adjacency matrices with different assortativity coefficients.
        Potentially several realisation of each (N_ensemble > 1)
        [(N_ensemble), r, N, N]
    r_stack : array_like, 1D float
        Stack of assortativity coefficients corresponding to A_stack.
    k_in : array_like, 1D int
        In-degree space.
    m : int
        Order of reconstruction. After applying Singular Value Decomposition
        the first m functions and singular values will be processed for
        reconstruction.

    Returns
    -------
    u : array_like, 4D float
        SVD basis functions (left).
        [Ensemble, Order, AssortativitySpace, DegreeSpace]
    v : array_like, 4D float
        SVD basis functions (right).
        [Ensemble, Order, AssortativitySpace, DegreeSpace]
    s : array_like, 3D float
        SVD singular values.
        [Ensemble, Order, AssortativitySpace]
    """

    if len(A_stack.shape) == 4:
        N_ensemble = A_stack.shape[0]
    else:
        N_ensemble = 1

    u = np.zeros((N_ensemble, m, len(r_stack), len(k_in)))
    v = np.zeros((N_ensemble, m, len(r_stack), len(k_in)))
    s = np.zeros((N_ensemble, m, len(r_stack)))
    for n in range(N_ensemble):
        for r in range(len(r_stack)):
            E = tn.generate.a_func_transform(A_stack[n, r], k_in)[0]
            u_nr, s_nr, vh_nr = np.linalg.svd(E)
            u[n, :, r, :] = u_nr[:, :m].T
            v[n, :, r, :] = vh_nr.T[:, :m].T
            s[n, :, r] = s_nr[:m]

    return u, v, s
