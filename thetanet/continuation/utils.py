import numpy as np


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
