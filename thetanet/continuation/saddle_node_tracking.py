import thetanet as tn
from thetanet.exceptions import *
from thetanet.continuation.utils import *
import time


def saddle_node(pm, init_b, init_x, init_y):
    """
    Pseudo arc-length saddle-node bifurcation continuation scheme.
    A saddle-node which occurred when varying x (pm.c_var) will be continued
    under the variation of y (pm.c_var2).
    Initial conditions need to be close to a saddle-node such that Newton's
    can converge towards the intended solution.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init_b : ndarray, 2D float
        Initial conditions for states.
    init_x : float
        Initial condition for first continuation parameter - will be treated
        like a parameter.
    init_y: float
        Initial condition for second continuation parameter - pseudo arc-length
        continuation will be done on that one.

    Returns
    -------
    b : ndarray, 3D complex
        Curve of fixed points. [Curve progress, states(2D)]
    x : ndarray, 1D float
        Curve of first continuation parameters.
    y : ndarray, 1D float
        Curve of second continuation parameters.
    """

    # Number of state variables
    N = tn.dynamics.degree_network.NQ_for_approach(pm)[0]

    # Precision for difference quotient
    dh1 = 1e-4  # general purpose
    dh2 = 1e-8  # for outer difference in directional derivative

    # Minimal step size when algorithm should stop
    ds_min = 0.1 * abs(pm.c_ds)

    # Initialise continuation
    init_b = real_stack(init_b)
    bnxy = np.zeros((pm.c_steps + 1, 2 * 2 * N + 2))
    bnxy[0, :2*N] = init_b
    bnxy[0, -2] = init_x
    bnxy[0, -1] = init_y

    # Determine dynamical equation
    dyn = init_dyn_sn(pm)

    # Determine eigenvector with eigenvalue with zero real part
    df_db = f_partial_b_sn(dyn, init_b, init_x, init_y, dh=dh1)
    df_dx = f_partial_x_sn(dyn, init_b, init_x, init_y, dh=dh1)
    init_n = np.linalg.solve(df_db, -df_dx)
    init_n /= np.linalg.norm(init_n)
    bnxy[0, 2*N:4*N] = init_n

    # Converge with Newton method to saddle-point
    for n_i in range(pm.c_n):
        j = jacobian_sn(dyn, bnxy[0], dh1=dh1, dh2=dh2)
        g = dyn_sn(dyn, bnxy[0], j[:2 * N, :2 * N])
        bnxy[0, :-1] -= np.linalg.solve(j[:, :-1], g)

    # To get the scheme started compute an initial null vector - the direction
    # of the first step. This should be done such that the change in c_var is
    # of the same sign as c_ds.
    null = null_vector(j)
    if np.sign(null[-1]) < 0:
        null *= -1

    print('Network with', N, 'degrees | Saddle-Node-Continuation of',
          pm.c_steps, 'steps:')
    t_start = time.time()
    try:
        for i in range(1, pm.c_steps + 1):
            do_newton = True
            while do_newton:

                # Step forward along null vector
                bnxy[i] = bnxy[i - 1] + null * pm.c_ds

                # Converge back to stability with Newton scheme
                for n_i in range(pm.c_n):
                    j = jacobian_sn(dyn, bnxy[i], dh1=dh1, dh2=dh2)
                    g = dyn_sn(dyn, bnxy[i], j[:2 * N, :2 * N])
                    # Constrain on y to ensure the solution stays on plane
                    # orthogonal to null vector and distance ds.
                    y_constrain = (bnxy[i] - bnxy[i - 1]).dot(null) - pm.c_ds
                    n_step = newton_step_sn(j, null, g, y_constrain)

                    bnxy[i] -= n_step

                    # if Newton's method is exact enough break
                    if np.abs(n_step).sum() < 1e-5:
                        do_newton = False
                        break

                # Adaptive step size - depending on speed of convergence
                if n_i < 2:
                    pm.c_ds *= 1.5
                if n_i > 3:
                    pm.c_ds /= 1.5

                if abs(pm.c_ds) < ds_min:
                    print("\nStep", i, "did not converge using minimal stepsize"
                                       " of " + str(ds_min) + "!")
                    raise ConvergenceError(i)

            # Update null vector and make sure direction is roughly the same as
            # previous one
            null_dummy = null_vector(j)
            if np.dot(null_dummy, null) > 0:
                null = null_dummy
            else:
                null = (-1) * null_dummy

            process = i / pm.c_steps
            tn.utils.progress_bar(process, time.time() - t_start)

    except (ConvergenceError) as error:
        final_step, = error.args
        bnxy = bnxy[:final_step]
        pass

    b, x, y = comp_unit(bnxy[:, :-2].T).T, bnxy[:, -2], bnxy[:, -1]

    return b, x, y


def dyn_sn(dyn, bnxy, j):
    """
    Compute the set of dynamical equations for saddle-node tracking.

       dyn(b,x,y) = 0  # dynamical equation at b
            j @ n = 0  # j being the jacobian j = df/db, n being the null vector
          n^2 - 1 = 0  # length of null vector

    Note that the jacobian computed in the scheme includes more derivatives
    and only the upper left corner is df/db.

    Parameters
    ----------
    dyn : function
        Dynamical equation.
    bnxy : ndarray, 1D float
        Continuation vector containing
        [b (2*N real states), n (2*N null vector), x (pm.c_var), y (pm.c_var2)]
    j : ndarray, 2D float
        Jacobian d(dyn(b))/db

    Returns
    -------
    g : ndarray, 1D float
        Dynamical equations for saddle-node continuation.
    """

    N = int((bnxy.shape[0]-2)/4)
    b, n, x, y = bnxy[: 2*N], bnxy[2*N:4*N], bnxy[-2], bnxy[-1]

    g = np.empty(4*N+1)
    g[:2*N] = real_stack(dyn(comp_unit(b), x, y))
    g[2*N:4*N] = j @ n
    g[-1] = n.dot(n) - 1

    return g


def init_dyn_sn(pm):
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
        setattr(locals()['pm'], pm.c_var, x)
        setattr(locals()['pm'], pm.c_var2, y)
        e = tn.utils.essential_fit(*pm.e_list, pm.r_list, pm.r)
        pm.usv = tn.utils.usv_from_essentials(*e, pm.c_in, pm.c_out)
        Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
        args = (0, b, pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                pm.kappa, pm.k_mean)
        if pm.c_pmap:
            return poi_map(*args)-b
        else:
            return dyn_equ(*args)

    return dyn


def jn_partial_b(f, b, n, x, y, dh1=1e-6, dh2=1e-6):
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

    j_hn = f_partial_b_sn(f, b + dh2 * n, x, y, dh=dh1)
    j = f_partial_b_sn(f, b, x, y, dh=dh1)
    djn_db = (j_hn - j) / dh2

    return djn_db


def jn_partial_x(f, b, n, x, y, dh1=1e-6, dh2=1e-6):
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

    j_hn = f_partial_x_sn(f, b + dh2 * n, x, y, dh=dh1)
    j = f_partial_x_sn(f, b, x, y, dh=dh1)
    djn_dx = (j_hn - j) / dh2

    return djn_dx


def jn_partial_y(f, b, n, x, y, dh1=1e-6, dh2=1e-6):
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

    j_hn = f_partial_y_sn(f, b + dh2 * n, x, y, dh=dh1)
    j = f_partial_y_sn(f, b, x, y, dh=dh1)
    djn_dy = (j_hn - j) / dh2

    return djn_dy


def f_partial_b_sn(f, b, x, y, dh=1e-6):
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


def f_partial_x_sn(f, b, x, y, dh=1e-6):
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


def f_partial_y_sn(f, b, x, y, dh=1e-6):
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


def jacobian_sn(f, bnxy, dh1=1e-6, dh2=1e-6):
    """ Compute Jacobi matrix containing the derivatives of f with respect to
    all states and the continuation variables x and y. Further more the
    derivatives of the equations defined in dyn_saddle_node.

    Parameters
    ----------
    f : function
        Dynamical equation.
    b_x : ndarray, 1D float
        States(real), variable.

    Returns
    -------
    j : ndarray, 2D float
        Jacobi matrix.(b is a real_stack)
        [[ df_db,    df_dn,     df_dx,     df_dy    ],
         [djn_db,    djn_dn,    djn_dx,    djn_dy   ],
         [d|n|-1_db, d|n|-1_dn, d|n|-1_dx, d|n|-1_dy]]
         =
        [[ df_db,    0,         df_dx,     df_dy    ],
         [djn_db,    df_db,     djn_dx,    djn_dy   ],
         [0,         2*n,       0,         0        ]]
    """

    N = int((bnxy.shape[0] - 2) / 4)
    b, n, x, y = bnxy[: 2 * N], bnxy[2 * N: 4 * N], bnxy[-2], bnxy[-1]
    df_db = f_partial_b_sn(f, b, x, y, dh=dh1)
    df_dx = f_partial_x_sn(f, b, x, y, dh=dh1)
    df_dy = f_partial_y_sn(f, b, x, y, dh=dh1)

    djn_db = jn_partial_b(f, b, n, x, y, dh1=dh1, dh2=dh2)
    djn_dx = jn_partial_x(f, b, n, x, y, dh1=dh1, dh2=dh2)
    djn_dy = jn_partial_y(f, b, n, x, y, dh1=dh1, dh2=dh2)

    j = np.zeros((4*N+1, 4*N+2))
    j[:2*N, :2*N] = df_db
    j[:2*N, -2] = df_dx
    j[:2*N, -1] = df_dy
    j[2*N:4*N, :2*N] = djn_db
    j[2*N:4*N, 2*N:4*N] = df_db
    j[2*N:4*N, -2] = djn_dx
    j[2*N:4*N, -1] = djn_dy
    j[-1, 2*N:4*N] = 2*n

    return j


def newton_step_sn(j, null, f, y_constrain):
    """
    Stepping down a gradient in Newton-method fashion and enforcing a
    constrain on y.

    Parameters
    ----------
    j : ndarray, 2D float
        Jacobi matrix.
    null : ndarray, 1D float
        Null vector of j.
    f : ndarray, 1D float
        Dynamical direction (dyn(bnxy)).
    y_constrain : float
        Constrain on y to ensure the solution stays on plane orthogonal to
        null vector and distance ds.

    Returns
    -------
    n_step : ndarray, 1D float
        Newton step.
    """

    m = np.append(j, null[None, :], 0)  # add null as last row
    n_step = np.linalg.solve(m, np.append(f, y_constrain))

    return n_step