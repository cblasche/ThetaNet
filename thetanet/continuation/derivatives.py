from thetanet.continuation.utils import *


def f_partial_b_1(f, b_x, dh=1e-6):
    """
    Compute partial derivative of f with respect to b.
    The states b come in real_stack version and need to be made complex first.
    The derivative is computed as a simple difference quotient.

    Parameters
    ----------
    f : function
        Dynamical equation.
    b_x : ndarray, 1D float
        States(real), variable.
    dh : float
        Infinitesimal step size.

    Returns
    -------
    df_db : ndarray, 2D float
        Partial derivative with respect to b.(real)
        [[df1_db1, df1_db2, ..., df1_dbn],
         [df2_db1, df2_db2, ..., df2_dbn],
          ...      ...      ...  ...
         [dfn_db1, dfn_db2, ..., dfn_dbn]]
    """

    b, x = b_x[:-1], b_x[-1]

    b = np.outer(b, np.ones(b.shape[0]))
    h = np.diag(np.full(b.shape[0], dh))
    b_h = b + h

    b = comp_unit(b[:, 0])
    b_h = comp_unit(b_h)

    # Try to process matrix at once - much faster. For poincare maps (using
    # scipy.integrate.solve_ivp internally) this can't be done. In this case
    # slice the matrix and process column wise.
    fb = f(b, x)[:, None] * np.ones(b_h.shape)
    try:
        df_db = (f(b_h, x) - fb) / dh
    except ValueError:
        df_db = np.zeros(b_h.shape, dtype=complex)
        for col in range(df_db.shape[1]):
            df_db[:, col] = (f(b_h[:, col], x) - fb[:, col]) / dh
    df_db = real_stack(df_db)

    return df_db


def f_partial_x_1(f, b_x, dh=1e-6):
    """
    Compute partial derivative of f with respect to variable x.
    The states b come in real_stack version and need to be made complex first.
    The derivative is computed as a simple difference quotient.

    Parameters
    ----------
    f : function
        Dynamical equation.
    b_x : ndarray, 1D float
        States(real), variable.
    dh : float
        Infinitesimal step size.

    Returns
    -------
    df_dx : ndarray, 1D float
        Partial derivative with respect to x.(real)
        [df1_dx, df2_dx, ..., dfn_dx]
    """

    b, x = b_x[:-1], b_x[-1]

    b = comp_unit(b)
    df_dx = (f(b, x+dh) - f(b, x)) / dh
    df_dx = real_stack(df_dx)

    return df_dx


def f_partial_b_2(f, b, x, y, dh=1e-6):
    """
    Compute partial derivative of f with respect to b.
    The states b come in real_stack version and need to be made complex first.
    The derivative is computed as a simple difference quotient.

    f : function
        Dynamical equation.
    b : ndarray, 1D float
        States(real), variable.
    x : ndarray, float
        Curve of first continuation parameters.
    y : ndarray, float
        Curve of second continuation parameters.
    dh : float
        Infinitesimal step size.

    Returns
    -------
    df_db : ndarray, 2D float
        Partial derivative with respect to b.(real)
        [[df1_db1, df1_db2, ..., df1_dbn],
         [df2_db1, df2_db2, ..., df2_dbn],
          ...      ...      ...  ...
         [dfn_db1, dfn_db2, ..., dfn_dbn]]
    """

    b = np.outer(b, np.ones(b.shape[0]))
    h = np.diag(np.full(b.shape[0], dh))
    b_h = b + h

    b = comp_unit(b[:, 0])
    b_h = comp_unit(b_h)

    # Try to process matrix at once - much faster. For poincare maps (using
    # scipy.integrate.solve_ivp internally) this can't be done. In this case
    # slice the matrix and process column wise.
    fb = f(b, x, y)[:, None] * np.ones(b_h.shape)
    try:
        df_db = (f(b_h, x, y) - fb) / dh
    except ValueError:
        df_db = np.zeros(b_h.shape, dtype=complex)
        for col in range(df_db.shape[1]):
            df_db[:, col] = (f(b_h[:, col], x, y) - fb[:, col]) / dh
    df_db = real_stack(df_db)

    return df_db


def f_partial_x_2(f, b, x, y, dh=1e-6):
    """
    Compute partial derivative of f with respect to variable x.
    The states b come in real_stack version and need to be made complex first.
    The derivative is computed as a simple difference quotient.

    f : function
        Dynamical equation.
    b : ndarray, 1D float
        States(real), variable.
    x : ndarray, float
        Curve of first continuation parameters.
    y : ndarray, float
        Curve of second continuation parameters.
    dh : float
        Infinitesimal step size.

    Returns
    -------
    df_dx : ndarray, 1D float
        Partial derivative with respect to x.(real)
        [df1_dx, df2_dx, ..., dfn_dx]
    """

    b = comp_unit(b)
    df_dx = (f(b, x+dh, y) - f(b, x, y)) / dh
    df_dx = real_stack(df_dx)

    return df_dx


def f_partial_y_2(f, b, x, y, dh=1e-6):
    """ Compute partial derivative of f with respect to variable y.
    The states b come in real_stack version and need to be made complex first.
    The derivative is computed as a simple difference quotient.

    f : function
        Dynamical equation.
    b : ndarray, 1D float
        States(real), variable.
    x : ndarray, float
        Curve of first continuation parameters.
    y : ndarray, float
        Curve of second continuation parameters.
    dh : float
        Infinitesimal step size.

    Returns
    -------
    df_dx : ndarray, 1D float
        Partial derivative with respect to y.(real)
        [df1_dy, df2_dy, ..., dfn_dy]
    """

    b = comp_unit(b)
    df_dy = (f(b, x, y+dh) - f(b, x, y)) / dh
    df_dy = real_stack(df_dy)

    return df_dy


def jn_partial_b_2(f, b, n, x, y, dh1=1e-6, dh2=1e-6):
    """
    Compute directional (n) derivative of jacobian of f with respect to b at
    parameter values x and y.

        d/db (j(b, x, y) @ n)

    Parameters
    ----------
    f : function
        Dynamical equation.
    b : ndarray, 1D float
        States(real), variable.
    n : ndarray, 1D float
        Null vector corresponding to zero real part eigenvalue.
    x : ndarray, float
        Curve of first continuation parameters.
    y : ndarray, float
        Curve of second continuation parameters.
    dh1 : float
        Infinitesimal step size for jacobian.
    dh2 : float
        Infinitesimal step size for directional derivative.

    Returns
    -------
    djn_db : ndarray, 2D float
        Directional derivative.
    """

    j_hn = f_partial_b_2(f, b + dh2 * n, x, y, dh=dh1)
    j = f_partial_b_2(f, b, x, y, dh=dh1)
    djn_db = (j_hn - j) / dh2

    return djn_db


def jn_partial_x_2(f, b, n, x, y, dh1=1e-6, dh2=1e-6):
    """
    Compute directional (n) derivative of jacobian of f with respect to x at
    parameter values x and y.

        d/dx (j(b, x, y) @ n)

    Parameters
    ----------
    f : function
        Dynamical equation.
    b : ndarray, 1D float
        States(real), variable.
    n : ndarray, 1D float
        Null vector corresponding to zero real part eigenvalue.
    x : ndarray, float
        Curve of first continuation parameters.
    y : ndarray, float
        Curve of second continuation parameters.
    dh1 : float
        Infinitesimal step size for jacobian.
    dh2 : float
        Infinitesimal step size for directional derivative.

    Returns
    -------
    djn_dx : ndarray, 2D float
        Directional derivative.
    """

    j_hn = f_partial_x_2(f, b + dh2 * n, x, y, dh=dh1)
    j = f_partial_x_2(f, b, x, y, dh=dh1)
    djn_dx = (j_hn - j) / dh2

    return djn_dx


def jn_partial_y_2(f, b, n, x, y, dh1=1e-6, dh2=1e-6):
    """
    Compute directional (n) derivative of jacobian of f with respect to y at
    parameter values x and y.

        d/dy (j(b, x, y) @ n)

    Parameters
    ----------
    f : function
        Dynamical equation.
    b : ndarray, 1D float
        States(real), variable.
    n : ndarray, 1D float
        Null vector corresponding to zero real part eigenvalue.
    x : ndarray, float
        Curve of first continuation parameters.
    y : ndarray, float
        Curve of second continuation parameters.
    dh1 : float
        Infinitesimal step size for jacobian.
    dh2 : float
        Infinitesimal step size for directional derivative.

    Returns
    -------
    djn_dy : ndarray, 2D float
        Directional derivative.
    """

    j_hn = f_partial_y_2(f, b + dh2 * n, x, y, dh=dh1)
    j = f_partial_y_2(f, b, x, y, dh=dh1)
    djn_dy = (j_hn - j) / dh2

    return djn_dy
