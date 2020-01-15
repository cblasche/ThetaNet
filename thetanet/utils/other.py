import numpy as np


def progress_bar(progress, time):
    """ Print progress bar to console output in the format
    Progress: [######### ] 90.0% in 10.22 sec

    Parameters
    ----------
    progress : float
        Value between 0 and 1.
    time : float
        Elapsed time till current progress.
    """

    print("\r| Progress: [{0:10s}] {1:.1f}% in {2:.0f} sec".format(
        '#' * int(progress * 10), progress * 100, time), end='')
    if progress >= 1:
        print("\r| Progress: [{0:10s}] {1:.1f}% in {2:.2f} sec".format(
            '#' * int(progress * 10), progress * 100, time))

    return


def three_term_recursion(N_mu, k):
    """ ThreeTermRecursion by Stiltjes.
    Constructing an orthonormal basis using Three Term Recursion method.
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
        A = np.sum(k * q[deg - 1] ** 2) / sum(q[deg - 1] ** 2)
        # first case is special: avoid division by zero
        if deg == 1:
            q[deg] = (k - A) * q[deg - 1]
            J[deg-1, deg-1] = A
        else:
            B = sum(q[deg-1] ** 2) / sum(q[deg - 2] ** 2)
            q[deg] = (k - A) * q[deg - 1] - B * q[deg - 2]
            J[deg - 1, deg - 1] = A
            J[deg - 2, deg - 1] = np.sqrt(B)
            J[deg - 1, deg - 2] = J[deg - 2, deg - 1]

    # normalising polynomials
    for deg in range(0, N_mu + 1):
            q[deg] /= np.linalg.norm(q[deg])

    # calculating abscissas k_v and weights w (Monien09 / numrecipes)
    k_v, psi = np.linalg.eigh(J)
    w = psi[0]**2 * (k.max()-k.min())

    return k_v, w, q
