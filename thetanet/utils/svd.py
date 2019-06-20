from scipy import interpolate
import numpy as np
import thetanet as tn


def usv_from_E(E, m=3):
    """ Decomposing E with SVD: E = u @ s @ v.T.
    Return those basis function and singular values s up to order m.

    Parameters
    ----------
    E : ndarray, 2D float
        Connectivity matrix for degrees flattened of size
        (N_c_in*N_c_out, N_c_in*N_c_out)
    m : int
        Rank of approximation.

    Returns
    -------
    u : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    """
    u, s, vh = np.linalg.svd(E)

    u = u[:, :m].T
    v = vh.T[:, :m].T
    s = s[:m]

    return u, s, v


def essentials_from_usv(u, s, v, c_in, c_out, deg_k):
    """ Fit polynomials of order deg_k to basis functions u and v.
    Return those coefficients plus singular values s.

    Parameters
    ----------
    u : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    c_in : ndarray, 1D float
        Cluster in-degrees representative.
    c_out : ndarray, 1D float
        Cluster out-degrees representative.
    deg_k : int
        Order of polynomial fit.

    Returns
    -------
    u_coeff : ndarray, 2D float
        Coefficients of polynomial fit.
        Size (m, number of coefficients for deg_k)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v_coeff : ndarray, 2D float
        Coefficients of polynomial fit.
        Size (m, number of coefficients for deg_k)
    """

    def coeff_from_basis_func(f, c_in, c_out, deg_k=3):
        """ Fit basis function with polynomial of order deg_k and return
        coefficients.

        Parameters
        ----------
        f : ndarray, 1D or 2D float
            Basis function or stack of m basis functions.
        c_in : ndarray, 1D float
            Cluster in-degrees representative.
        c_out : ndarray, 1D float
            Cluster out-degrees representative.
        deg_k : int
            Order of polynomial fit.

        Returns
        -------
        u : ndarray, 2D float
            Coefficients of polynomial fit.
            Size (m, number of coefficients for deg_k)
        """
        if len(f.shape) == 2:
            m = 1
        if len(f.shape) == 3:
            m = f.shape[0]

        A = polynomials_2d(c_in, c_out, deg_k)
        A = A.reshape(A.shape[0], -1).T

        coeff = [None] * m
        for m_i in range(m):
            coeff[m_i], r, rank, s = np.linalg.lstsq(A, f[m_i].flatten(),
                                                     rcond=None)
        return np.asarray(coeff)

    N_c_in = len(c_in)
    N_c_out = len(c_out)

    u.shape = (-1, N_c_in, N_c_out)
    v.shape = u.shape

    u_coeff = coeff_from_basis_func(u, c_in, c_out, deg_k)
    v_coeff = coeff_from_basis_func(v, c_in, c_out, deg_k)

    return u_coeff, s, v_coeff


def polynomials_2d(c_in, c_out, deg_k=3):
    """ 2d polynomials flattened on [c_in]x[c_out] grid with total number
    of order up to 'deg'.

    Parameters
    ----------
    c_in : ndarray, 1D float
        Representative degree of each cluster.
    c_out : ndarray, 1D float
        Representative degree of each cluster.
    deg_k : int
        Highest order polynomial.

    Returns
    -------
    A : ndarray, 3D float [polynomial, c_in, c_out]
        List of polynomials.
    """
    # to shift coefficients in a reasonable range divide by k_mean
    k_mean = np.append(c_in, c_out).mean()
    X, Y = np.meshgrid(c_in, c_out) / k_mean

    if deg_k == 1:
        A = np.array([X ** 0,
                      X, Y])
    if deg_k == 2:
        A = np.array([X ** 0,
                      X, Y,
                      X ** 2, X * Y, Y ** 2])
    if deg_k == 3:
        A = np.array([X ** 0,
                      X, Y,
                      X ** 2, X * Y, Y ** 2,
                      X ** 3, X ** 2 * Y, Y ** 2 * X, Y ** 3])
    if deg_k == 4:
        A = np.array([X ** 0,
                      X, Y,
                      X ** 2, X * Y, Y ** 2,
                      X ** 3, X ** 2 * Y, Y ** 2 * X, Y ** 3,
                      X ** 4, X ** 3 * Y, X ** 2 * Y ** 2, X * Y ** 3, Y ** 4])
    return A


def essential_list_from_data(A_list, r_list, N_c_in, N_c_out, deg_k=3, m=3,
                             mapping='cumsum'):
    """ For each transformed degree connectivity E of adjacency matrices
    in A_list compute the SVD up to rank m. Fit polynomials through basis
    functions and return them together with the singular values ('essentials').

    Parameters
    ----------
    A_list : ndarray, 3D int
        List of adjacency matrices with different values of assortativity.
    r_list : ndarray, 1D float
        List of respective assortativity coefficients.
    N_c_in : int
        Number of in-degree clusters
    N_c_out : int
        Number of out-degree clusters
    deg_k : int
        Order of polynomial fit.
    m : int
        Rank of approximation.
    mapping : str
        'linear' for equally sized cluster bins or
        'cumsum' for P_k adapted sized cluster bins for more evenly filled
         bins.

    Returns
    -------
    u_coeff_list : ndarray, 3D float
        List of coefficients of polynomial fit.
        Size (N_r, m, number of coefficients for deg_k)
    s_list : ndarray, 2D float
        List of singular Values. Size (N_r, m)
    v_coeff_list : ndarray, 3D float
        List of coefficients of polynomial fit.
        Size (N_r, m, number of coefficients for deg_k)
    """
    N_r = len(r_list)
    N_coeff_dict = {0: 1, 1: 3, 2: 6, 3: 10, 4: 15}
    N_coeff = N_coeff_dict[deg_k]

    u_coeff_list = np.empty((N_r, m, N_coeff))
    v_coeff_list = np.empty((N_r, m, N_coeff))
    s_list = np.empty((N_r, m))

    for i in range(N_r):
        E, B, c_in, c_out = tn.generate.a_func_transform(A_list[i], N_c_in,
                                                         N_c_out,
                                                         mapping=mapping)
        usv = usv_from_E(E, m)
        u_coeff_list[i], s_list[i], v_coeff_list[i] = \
            essentials_from_usv(*usv, c_in, c_out, deg_k)

    return u_coeff_list, s_list, v_coeff_list


def essential_fit(u_coeff_list, s_list, v_coeff_list, r_list, r):
    """ Given the list of coefficients and singular values for a series of
    r's in r_list: fit those values for the assortativity coefficient r.

    Parameters
    ----------
    u_coeff_list : ndarray, 3D float
        List of coefficients of polynomial fit.
        Size (N_r, m, number of coefficients for deg_k)
    s_list : ndarray, 2D float
        List of singular Values. Size (N_r, m)
    v_coeff_list : ndarray, 3D float
        List of coefficients of polynomial fit.
        Size (N_r, m, number of coefficients for deg_k)
    r_list : ndarray, 1D float
        Assortativity coefficients.
    r : float
        Desired assortativity coefficient.

    Returns
    -------
    u_coeff : ndarray, 2D float
        Coefficients of polynomial fit.
        Size (m, number of coefficients for deg_k)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v_coeff : ndarray, 2D float
        Coefficients of polynomial fit.
        Size (m, number of coefficients for deg_k)
    """
    u_coeff_func = interpolate.interp1d(r_list, np.moveaxis(u_coeff_list, 0,
                                                            -1), kind='cubic')
    u_coeff = u_coeff_func(r)

    s_func = interpolate.interp1d(r_list, s_list.T, kind='cubic')
    s = s_func(r)

    v_coeff_func = interpolate.interp1d(r_list, np.moveaxis(v_coeff_list, 0,
                                                            -1), kind='cubic')
    v_coeff = v_coeff_func(r)

    return u_coeff, s, v_coeff


def usv_from_essentials(u_coeff, s, v_coeff, c_in, c_out):
    """ Reconstruct basis functions from coefficients and for symmetry reasons
    pass s through this function.

    Parameters
    ----------
    u_coeff : ndarray, 2D float
        Coefficients of polynomial fit.
        Size (m, number of coefficients for deg_k)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v_coeff : ndarray, 2D float
        Coefficients of polynomial fit.
        Size (m, number of coefficients for deg_k)
    c_in : ndarray, 1D float
        Cluster in-degrees representative.
    c_out : ndarray, 1D float
        Cluster out-degrees representative.

    Returns
    -------
    u : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    """

    def basis_func_from_coeff(coeff, c_in, c_out):
        """ Reconstruct polynomial approximation of basis function.

        Parameters
        ----------
        coeff : ndarray, 2D float
            Coefficients of polynomial fit.
            Size (m, number of coeffcients for deg_k)
        c_in : ndarray, 1D float
            Cluster in-degrees representative.
        c_out : ndarray, 1D float
            Cluster out-degrees representative.

        Returns
        -------
        f : ndarray, 1D or 2D float
            Basis function or stack of m basis functions.
        """
        m = coeff.shape[0]
        deg_k_dict = {1: 0, 3: 1, 6: 2, 10: 3, 15: 4}
        deg_k = deg_k_dict[coeff.shape[1]]

        A = polynomials_2d(c_in, c_out, deg_k)

        f = np.zeros((m, np.prod(A.shape[1:])))
        for m_i in range(m):
            f[m_i] = np.tensordot(coeff[m_i], A, axes=(0, 0)).flatten()

        return f

    u = basis_func_from_coeff(u_coeff, c_in, c_out)
    v = basis_func_from_coeff(v_coeff, c_in, c_out)

    return u, s, v


def E_from_usv(u, s, v):
    """ Reconstruct E from singular values and basis functions.

    Parameters
    ----------
    u : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)
    s : ndarray, 1D float
        Singular Values. Size (m)
    v : ndarray, 2D float
        Basis function. Size (m, N_c_in*N_c_out)

    Returns
    -------
    E : ndarray, 2D float
        Reconstructed connectivity matrix for degrees flattened of size
        (N_c_in*N_c_out, N_c_in*N_c_out)
    """

    E = (u.T * s) @ v

    return E
