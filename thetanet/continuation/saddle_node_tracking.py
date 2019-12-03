import thetanet as tn
from thetanet.exceptions import *
from thetanet.continuation import *
import time


def saddle_node(pm, init_b=None, init_x=None, init_y=None, init_full_state=None,
                full_state_output=False, adaptive_step_size=True, dh1=1e-7,
                dh2=1e-5):
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
    init_full_state : ndarray, 1D float
        Full initial condition bnxy including null vector.
    full_state_output : bool
        Return the full bnxy array if True.
    adaptive_step_size: bool
        Adjust step size pm.c_ds according to speed of convergence if True.
    dh1 : float
        Precision when building difference quotient.
    dh2 : float
        Precision when building outer difference quotient in d(j@n)_db.
        Needs to be bigger than dh1.

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

    # Minimal step size when algorithm should stop
    ds_min = 0.01 * abs(pm.c_ds)
    # Maximal step size which should not be exceeded
    ds_max = 100 * abs(pm.c_ds)

    # Determine dynamical equation
    dyn = init_dyn_2(pm)

    # Initialise continuation
    bnxy = np.zeros((pm.c_steps + 1, 2 * 2 * N + 2))
    if init_full_state is None:
        init_b = real_stack(init_b)
        # Determine eigenvector with eigenvalue with zero real part
        df_db = f_partial_b_2(dyn, init_b, init_x, init_y, dh=dh1)
        df_dx = f_partial_x_2(dyn, init_b, init_x, init_y, dh=dh1)
        init_n = np.linalg.solve(df_db, -df_dx)
        init_n /= np.linalg.norm(init_n)
        bnxy[0, :2*N] = init_b
        bnxy[0, 2*N:4*N] = init_n
        bnxy[0, -2] = init_x
        bnxy[0, -1] = init_y
    else:
        bnxy[0] = init_full_state

    # Converge with Newton method to saddle-point
    for n_i in range(pm.c_n):
        j = jacobian_sn(dyn, bnxy[0], dh1=dh1, dh2=dh2)
        g = dyn_sn(dyn, bnxy[0], j[:2 * N, :2 * N])
        bnxy[0, :-1] -= np.linalg.solve(j[:, :-1], g)

    # To get the scheme started compute an initial null vector - the direction
    # of the first step. This should be done such that the change in c_var is
    # of the same sign as c_ds.
    j = jacobian_sn(dyn, bnxy[0], dh1=dh1, dh2=dh2)
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
                    n_step = newton_step(j, null, g, y_constrain)

                    bnxy[i] -= n_step

                    # if Newton's method is exact enough break
                    if np.abs(n_step).sum() < 1e-4:
                        do_newton = False
                        break

                # Adaptive step size - depending on speed of convergence
                if adaptive_step_size:
                    if n_i < 2:
                        pm.c_ds *= 1.5
                        pm.c_ds = np.clip(pm.c_ds, -ds_max, ds_max)
                    if n_i > 3:
                        pm.c_ds /= 1.5
                    if abs(pm.c_ds) < ds_min:
                        print("\nStep", i, "did not converge using minimal "
                                           "step size of " + str(ds_min) + "!")
                        raise ConvergenceError(i)
                else:
                    if do_newton:
                        print("\nStep", i, "did not converge!")
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

    if full_state_output:
        return bnxy
    else:
        b, x, y = comp_unit(bnxy[:, :2*N].T).T, bnxy[:, -2], bnxy[:, -1]
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


def jacobian_sn(f, bnxy, dh1=1e-7, dh2=1e-5):
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
    df_db = f_partial_b_2(f, b, x, y, dh=dh1)
    df_dx = f_partial_x_2(f, b, x, y, dh=dh1)
    df_dy = f_partial_y_2(f, b, x, y, dh=dh1)

    djn_db = jn_partial_b_2(f, b, n, x, y, dh1=dh1, dh2=dh2)
    djn_dx = jn_partial_x_2(f, b, n, x, y, dh1=dh1, dh2=dh2)
    djn_dy = jn_partial_y_2(f, b, n, x, y, dh1=dh1, dh2=dh2)

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
