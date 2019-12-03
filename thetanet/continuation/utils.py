import numpy as np
import thetanet as tn


def null_vector(A):
    """
    Compute normalised null vector to a given matrix A.
    The null vector fulfills A(null) = 0.

    Parameters
    ----------
    A : ndarray, 2D float
        Matrix.

    Returns
    -------
    null : ndarray, 1D float
        Null vector, normalised.
    """

    null = nullspace(A)[:, -1]
    null /= (np.sqrt(np.dot(null, null)))

    return null


def nullspace(A, atol=1e-14, rtol=0):
    """
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return ns


def real_stack(x):
    """
    Split a complex variable x into two parts and stack it to have a twice
    twice as long real variable.

    Parameters
    ----------
    x : ndarray, complex

    Returns
    -------
    x : ndarray, real
    """
    return np.append(x.real, x.imag, axis=0)


def comp_unit(x):
    """
    Reverse process of real_stack. Add second half of variable x as
    imaginary part to real first half.

    Parameters
    ----------
    x : ndarray, real

    Returns
    -------
    x : ndarray, complex
    """
    return x[:int(x.shape[0]/2)] + 1j * x[int(x.shape[0]/2):]


def newton_step(j, null, g, y_constrain):
    """
    Stepping down a gradient in Newton-method fashion and enforcing a
    constrain on y.

    Parameters
    ----------
    j : ndarray, 2D float
        Jacobi matrix.
    null : ndarray, 1D float
        Null vector of j.
    g : ndarray, 1D float
        Set of dynamical equations evaluated at current position.
    y_constrain : float
        Constrain on y to ensure the solution stays on plane orthogonal to
        null vector and distance ds.

    Returns
    -------
    n_step : ndarray, 1D float
        Newton step.
    """

    m = np.append(j, null[None, :], 0)  # add null as last row
    n_step = np.linalg.solve(m, np.append(g, y_constrain))

    return n_step


def init_dyn_1(pm):
    """
    Depending on the choice of degree approach and continuation variable
    the dynamical equations might need some other parameters updated.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.

    Returns
    -------
    dyn : function
        Dynamical equation.
    """

    from thetanet.dynamics.degree_network import dynamical_equation \
        as dyn_equ
    from thetanet.dynamics.degree_network import poincare_map as poi_map

    def dyn(b, x):
        setattr(pm, pm.c_var, x)
        if pm.c_var == 'rho':
            if pm.degree_approach == 'virtual':
                pm.w = pm.w_func(pm.rho)
            elif pm.degree_approach == 'transform':
                pm.usv = pm.usv_func(pm.rho)
        if pm.c_var == 'r':
            if pm.degree_approach == 'virtual':
                pm.a_v = pm.a_v_func(pm.r)
            elif pm.degree_approach == 'transform':
                pm.usv = pm.usv_func(pm.r)
        Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
        args = (0, b, pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                pm.kappa, pm.k_mean)
        if pm.c_pmap:
            return poi_map(*args)-b
        else:
            return dyn_equ(*args)

    return dyn


def init_dyn_2(pm):
    """ Depending on the choice of degree approach and continuation variables
    the dynamical equations might need some other parameters updated.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.

    Returns
    -------
    dyn : function
        Dynamical equation.
    """

    from thetanet.dynamics.degree_network import dynamical_equation \
        as dyn_equ
    from thetanet.dynamics.degree_network import poincare_map as poi_map

    def dyn(b, x, y):
        setattr(pm, pm.c_var, x)
        setattr(pm, pm.c_var2, y)
        if pm.c_var == 'rho' or pm.c_var2 == 'rho':
            if pm.degree_approach == 'virtual':
                pm.w = pm.w_func(pm.rho)
            elif pm.degree_approach == 'transform':
                pm.usv = pm.usv_func(pm.rho)
        if pm.c_var == 'r' or pm.c_var2 == 'r':
            if pm.degree_approach == 'virtual':
                pm.a_v = pm.a_v_func(pm.r)
            elif pm.degree_approach == 'transform':
                pm.usv = pm.usv_func(pm.r)
        Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
        args = (0, b, pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                pm.kappa, pm.k_mean)
        if pm.c_pmap:
            return poi_map(*args)-b
        else:
            return dyn_equ(*args)

    return dyn

