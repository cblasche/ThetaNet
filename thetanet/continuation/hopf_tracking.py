import thetanet as tn
from thetanet.exceptions import *
from thetanet.continuation import *
import time


def hopf(pm, init_b=None, init_x=None, init_y=None, init_full_state=None,
         full_state_output=False, adaptive_step_size=True, dh1=1e-7, dh2=1e-5):
    """
    Pseudo arc-length hopf bifurcation continuation scheme.
    A hopf bifurcation which occurred when varying x (pm.c_var) will be
    continued under the variation of y (pm.c_var2).
    Initial conditions need to be close to a hopf such that Newton's method
    can converge towards the intended solution.

    We denote the eigenvector with zero real part and a complex pair +/-o as
    eigenvalues as (c + j*d).

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
        Full initial condition bcdoxy including eigenvector and -value of zero
        real part eigenvalue.
    full_state_output : bool
        Return the full bcdoxy array if True.
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
    ds_min = 1e-3 * abs(pm.c_ds)
    # Maximal step size which should not be exceeded
    ds_max = 1e2 * abs(pm.c_ds)

    # Determine dynamical equation
    dyn = init_dyn_2(pm)

    # Initialise continuation
    bcdoxy = np.zeros((pm.c_steps + 1, 3 * 2 * N + 3))

    if init_full_state is None:
        # Determine eigenvector (c+j*d) and eigenvalue o with maximal real part
        init_b = real_stack(init_b)
        df_db = f_partial_b_2(dyn, init_b, init_x, init_y, dh=dh1)
        val, vec = np.linalg.eig(df_db)
        o_ind = np.argmax(val.real)
        init_o = val[o_ind].imag
        init_c, init_d = vec[:, o_ind].real, vec[:, o_ind].imag

        bcdoxy[0, :2*N] = init_b
        bcdoxy[0, 2*N:4*N] = init_c
        bcdoxy[0, 4*N:6*N] = init_d
        bcdoxy[0, -3] = init_o
        bcdoxy[0, -2] = init_x
        bcdoxy[0, -1] = init_y

        phi = init_c / np.linalg.norm(init_c)**2

        # Converge with Newton method on solution curve
        for n_i in range(pm.c_n):
            j = jacobian_hopf(dyn, bcdoxy[0], phi, dh1=dh1, dh2=dh2)
            g = dyn_hopf(dyn, bcdoxy[0], j[:2*N, :2*N], phi)
            bcdoxy[0, :-1] -= np.linalg.solve(j[:, :-1], g)

    else:
        bcdoxy[0] = init_full_state
        init_c = bcdoxy[0, 2 * N:4 * N]
        phi = init_c / np.linalg.norm(init_c) ** 2

    # To get the scheme started compute an initial null vector - the direction
    # of the first step. This should be done such that the change in c_var is
    # of the same sign as c_ds.
    j = jacobian_hopf(dyn, bcdoxy[0], phi, dh1=dh1, dh2=dh2)
    null = null_vector(j)
    if np.sign(null[-1]) < 0:
        null *= -1

    print('Network with', N, 'degrees | Hopf-Continuation of',
          pm.c_steps, 'steps:')
    t_start = time.time()
    try:
        for i in range(1, pm.c_steps + 1):
            do_newton = True
            while do_newton:

                # Step forward along null vector
                bcdoxy[i] = bcdoxy[i - 1] + null * pm.c_ds

                # Converge back to stability with Newton scheme
                for n_i in range(pm.c_n):
                    j = jacobian_hopf(dyn, bcdoxy[i], phi, dh1=dh1, dh2=dh2)
                    g = dyn_hopf(dyn, bcdoxy[i], j[:2 * N, :2 * N], phi)
                    # Constrain on y to ensure the solution stays on plane
                    # orthogonal to null vector and distance ds.
                    y_constrain = (bcdoxy[i] - bcdoxy[i - 1]).dot(null) - \
                                  pm.c_ds
                    n_step = newton_step(j, null, g, y_constrain)

                    bcdoxy[i] -= n_step

                    # print(np.abs(n_step).sum())

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
        bcdoxy = bcdoxy[:final_step]
        pass

    if full_state_output:
        return bcdoxy
    else:
        b, x, y = comp_unit(bcdoxy[:, :2*N].T).T, bcdoxy[:, -2], bcdoxy[:, -1]
        return b, x, y


def dyn_hopf(dyn, bcdoxy, j, phi):
    """
    Compute the set of dynamical equations for hopf bifurcation tracking.
    Note:
    1) Jacobian j = df/db
    2) Eigenequation: j @ (c+j*d) = +/-i*o * (c+i*d)

       dyn(b,x,y) = 0  # dynamical equation at b
            j @ c = -o * d
            j @ d = o * c
       phi.dot(c) = 1
       phi.dot(d) = 0

    Note that the jacobian computed in the scheme includes more derivatives
    and only the upper left corner is df/db.

    Parameters
    ----------
    dyn : function
        Dynamical equation.
    bcdoxy : ndarray, 1D float
        Continuation vector containing
        [b (2*N real states),
         c (2*N real part eigenvector),
         d (2*N imag part eigenvector),
         o (imag eigenvalue),
         x (pm.c_var),
         y (pm.c_var2)]
    j : ndarray, 2D float
        Jacobian d(dyn(b))/db
    phi : ndarray, 1D float
        Constant vector to fix eigenvector.

    Returns
    -------
    g : ndarray, 1D float
        Dynamical equations for saddle-node continuation.
    """

    N = int((bcdoxy.shape[0]-3)/6)
    b, c, d = bcdoxy[: 2 * N], bcdoxy[2 * N: 4 * N], bcdoxy[4 * N: 6 * N]
    o, x, y = bcdoxy[-3], bcdoxy[-2], bcdoxy[-1]

    g = np.empty(6*N+2)
    g[:2*N] = real_stack(dyn(comp_unit(b), x, y))
    g[2*N:4*N] = j @ c + o * d
    g[4*N:6*N] = j @ d - o * c
    g[-2] = phi.dot(c) - 1
    g[-1] = phi.dot(d)

    return g


def jacobian_hopf(f, bcdoxy, phi, dh1=1e-6, dh2=1e-6):
    """ Compute Jacobi matrix containing the derivatives of f with respect to
    all states, the eigenvectors c,d, the eigenvalue o, and the continuation
    variables x and y.
    Further more the derivatives of the equations defined in dyn_hopf.

    Parameters
    ----------
    f : function
        Dynamical equation.
    bcdoxy : ndarray, 1D float
        States(real), variable.
    phi : ndarray, 1D float
        Constant vector to fix eigenvector.

    Returns
    -------
    j : ndarray, 2D float
        Jacobi matrix.(b is a real_stack)
        [[ df_db,    df_dc,     df_dd,     df_do,     df_dx,     df_dy    ],
         [djc_db,    djc_dc,    djc_dd,    djc_do,    djc_dx,    djc_dy   ],
         [djd_db,    djd_dc,    djd_dd,    djd_do,    djd_dx,    djd_dy   ],
         [d|c|-1_db, d|c|-1_dc, d|c|-1_dd, d|c|-1_do, d|c|-1_dx, d|c|-1_dy]]
         [d|d|_db,   d|d|_dc,   d|d|_dd,   d|d|_do,   d|d|_dx,   d|d|_dy  ]]
         =
        [[ df_db,    0,         0,         0,         df_dx,     df_dy    ],
         [djc_db,    df_db,     o,         d,         djc_dx,    djc_dy   ],
         [djd_db,    -o,        df_db,     -c,        djd_dx,    djd_dy   ],
         [0,         phi,       0,         0,         0,         0,       ]]
         [0,         0,         phi,       0,         0,         0,       ]]
    """

    N = int((bcdoxy.shape[0] - 3) / 6)
    b, c, d = bcdoxy[: 2 * N], bcdoxy[2 * N: 4 * N], bcdoxy[4 * N: 6 * N]
    o, x, y = bcdoxy[-3], bcdoxy[-2], bcdoxy[-1]

    df_db = f_partial_b_2(f, b, x, y, dh=dh1)
    df_dx = f_partial_x_2(f, b, x, y, dh=dh1)
    df_dy = f_partial_y_2(f, b, x, y, dh=dh1)

    djc_db = jn_partial_b_2(f, b, c, x, y, dh1=dh1, dh2=dh2)
    djc_dx = jn_partial_x_2(f, b, c, x, y, dh1=dh1, dh2=dh2)
    djc_dy = jn_partial_y_2(f, b, c, x, y, dh1=dh1, dh2=dh2)

    djd_db = jn_partial_b_2(f, b, d, x, y, dh1=dh1, dh2=dh2)
    djd_dx = jn_partial_x_2(f, b, d, x, y, dh1=dh1, dh2=dh2)
    djd_dy = jn_partial_y_2(f, b, d, x, y, dh1=dh1, dh2=dh2)

    j = np.zeros((6*N+2, 6*N+3))

    j[:2*N, :2*N] = df_db
    j[:2*N, -2] = df_dx
    j[:2*N, -1] = df_dy

    j[2*N:4*N, :2*N] = djc_db
    j[2*N:4*N, 2*N:4*N] = df_db
    j[2*N:4*N, 4*N:6*N] = np.diag(2*N*[o])
    j[2*N:4*N, -3] = d
    j[2*N:4*N, -2] = djc_dx
    j[2*N:4*N, -1] = djc_dy

    j[4*N:6*N, :2*N] = djd_db
    j[4*N:6*N, 2*N:4*N] = np.diag(2*N*[-o])
    j[4*N:6*N, 4*N:6*N] = df_db
    j[4*N:6*N, -3] = -c
    j[4*N:6*N, -2] = djd_dx
    j[4*N:6*N, -1] = djd_dy

    j[-2, 2*N:4*N] = phi
    j[-1, 4*N:6*N] = phi

    return j
