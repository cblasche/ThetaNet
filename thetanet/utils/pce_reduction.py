import numpy as np


def three_term_recursion(N_mu, k, P_k):
    """ ThreeTermRecursion by Stiltjes.
    Constructing an orthonormal basis with respect to weight function P_k
    using Three Term Recursion method.
    Highest order polynomial will be of order k^(N_mu).
    This function returns an 2D-array q[degree, k].

    P_(n+1) = (k - A_n) * P_n - B_n * P_(n-1)
    A_n     = < k*P_n, P_n > / < P_n, P_n >
    B_n     = < P_n, P_n > / < P_(n-1), P_(n-1) >

    Parameters
    ----------
    N_mu : int
        Number of roots of highest polynomial (k_v), hence there are
        N_mu+1 polynomials.
    k : ndarray, 1D int
        Degree space.
    P_k : ndarray, 1D float
        Degree probability.

    Returns
    -------
    k_v : ndarray, 1D float
        Virtual degrees / roots of highest polynomial.
    w : ndarray, 1D float
        Weights of virtual degrees.
    q : ndarray, 2D float
        Polynomials [degree, k].
    """

    q = np.zeros((N_mu + 1, len(k)))
    J = np.zeros((N_mu, N_mu))  # Jacobi Matrix: tri-diagonal matrix

    q[0] = k ** 0  # P_0 = 1
    for deg in range(1, N_mu + 1):
        A = np.sum(k * q[deg - 1] ** 2 * P_k) / sum(q[deg - 1] ** 2 * P_k)
        # first case is special: avoid division by zero
        if deg == 1:
            q[deg] = (k - A) * q[deg - 1]
            J[deg-1, deg-1] = A
        else:
            B = sum(q[deg-1] ** 2 * P_k) / sum(q[deg - 2] ** 2 * P_k)
            q[deg] = (k - A) * q[deg - 1] - B * q[deg - 2]
            J[deg - 1, deg - 1] = A
            J[deg - 2, deg - 1] = np.sqrt(B)
            J[deg - 1, deg - 2] = J[deg - 2, deg - 1]

    # normalising polynomials
    for deg in range(0, N_mu + 1):
        q[deg] /= np.sqrt(sum(q[deg] ** 2 * P_k))

    # calculating abscissas k_v and weights w (Monien09 / numrecipes)
    k_v, psi = np.linalg.eigh(J)
    w = psi[0]**2

    return k_v, w, q


def func_v(f, k, k_v):
    """ Evaluate a stack of functions f, which are defined on degree space k on
    the virtual degree space k_v. For single functions f use
    np.interp(k_v, k, f).

    Parameters
    ----------
    f : ndarray, 2D float
        Function stack to be 'virtualised' - interpolated on virtual degree space.
    k : ndarray, 1D int
        Degree space.
    k_v : ndarray, 1D float
        Virtual degree space.

    Returns
    -------
    f_v : ndarray, 2D float
        [polynomial degree (mu), function values of f at virtual degrees]
    """

    if f.shape[1] != k.shape[0]:
        f = f.T
        if f.shape[1] != k.shape[0]:
            print('Make sure function f lives on space k.')
            return

    N_mu = k_v.shape[0]
    N_stack = f.shape[0]
    f_v = np.zeros((N_stack, N_mu))
    for i in range(N_stack):
        f_v[i] = np.interp(k_v, k, f[i])

    return f_v
