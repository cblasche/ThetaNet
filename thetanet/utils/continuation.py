import numpy as np
import thetanet as tn
import time
from thetanet.exceptions import *


def continuation(pm, init_b=None, init_x=None, init_stability=None):
    """ Continuation scheme.
    Trace out a curve stable or unstable fixed points by continuing an initial
    solution under variation of the chosen variable pm.c_var.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init_b : ndarray, 2D float
        Initial conditions for states followed by continuation variable.
    init_x : float
        Initial condition for continuation parameter.
    init_stability: bool
        True if fixed point is stable, False if not.


    Returns
    -------
    b : ndarray, 3D complex
        Curve of fixed points. [Curve progress, states(2D)]
    x : ndarray, 1D float
        Curve of continuation parameters.
    stable : ndarray, 1D bool
        Curve of stability information of the respective point in b and x.
    """

    if pm.c_var not in pm.c_lib:
        print('\n "c_var" is none of the possible choices', pm.c_lib,
              '.')
        exit(1)

    N_state_variables, Q = tn.dynamics.degree_network.NQ_for_approach(pm)

    # Initialise continuation
    b_x = np.zeros((pm.c_steps + 1, 2 * N_state_variables + 1))

    if init_b is None:
        init_b = tn.dynamics.degree_network.integrate(pm)[-1]
        init_b.shape = N_state_variables
        # For poincare map: Integrate initial condition till poincare section
        # is reached.
        if pm.c_pmap:
            init_b = tn.dynamics.degree_network.poincare_map(0, init_b, pm.Gamma,
                                                             pm.n, pm.d_n, Q,
                                                             pm.eta_0, pm.delta,
                                                             pm.kappa, pm.k_mean)
        init_x = eval('pm.' + pm.c_var)
        init_stability = True

    try:
        init_b.shape = N_state_variables
    except ValueError:
        print('\n Invalid initial conditions! Check length of initial '
              'condition and chosen degree approach.')
        quit()

    b_x[0, :-1] = real_stack(init_b)
    b_x[0, -1] = init_x

    # Stability information of fixed points
    stable = np.zeros(pm.c_steps + 1)
    stable[0] = init_stability

    # Determine dynamical equation
    dyn = init_dyn(pm)

    # To get the scheme started compute an initial null vector - the direction
    # of the first step. This should be done such that the change in c_var is
    # of the same sign as c_ds.
    j = jacobian(dyn, b_x[0])
    null = null_vector(j)
    if np.sign(null[-1]) < 0:
        null *= -1

    print('Network with', N_state_variables, 'degrees | Continuation of',
          pm.c_steps, 'steps:')
    t_start = time.time()
    try:
        for i in range(1, pm.c_steps+1):
            # Step forward along null vector
            b_x[i] = b_x[i - 1] + null * pm.c_ds

            # Converge back to stability with Newton scheme
            for n_i in range(pm.c_n):
                try:
                    j = jacobian(dyn, b_x[i])
                except NoPeriodicOrbitException:
                    raise NoPeriodicOrbitException(i)
                db_x = real_stack(dyn(comp_unit(b_x[i, :-1]), b_x[i, -1]))
                # Constrain on x to ensure the solution stays on plane
                # orthogonal to null vector and distance ds.
                x_constrain = (b_x[i] - b_x[i-1]).dot(null) - pm.c_ds
                n_step = newton_step(j, null, db_x, x_constrain)

                b_x[i] = b_x[i] - n_step

                # if Newton's method is exact enough break
                if np.abs(n_step).sum() < 1e-5:
                    break
                if n_i == (pm.c_n - 1):
                    if i is 1 and not pm.c_pmap:
                        print("\nCheck if there is a periodic orbit and if so"
                              " use poincare map. (c_pmap=True)")
                        exit(1)
                    print("\nStep", i, "did not converge!")
                    raise ConvergenceError(i)

            # Adaptive step size - depending on speed of convergence
            if n_i < 2:
                pm.c_ds *= 1.5
            if n_i > 3:
                pm.c_ds /= 1.5

            # Stability (larges eigenvalue less than 0)
            stable[i] = np.max(np.linalg.eig(j[:, :-1])[0].real) < 0

            # Update null vector and make sure direction is roughly the same as
            # previous one
            null_dummy = null_vector(j)
            if np.dot(null_dummy, null) > 0:
                null = null_dummy
            else:
                null = (-1) * null_dummy

            process = i / pm.c_steps
            tn.utils.progress_bar(process, time.time() - t_start)

    except (ConvergenceError, NoPeriodicOrbitException) as error:
        final_step, = error.args
        b_x = b_x[:final_step]
        stable = stable[:final_step]
        pass

    b = comp_unit(b_x[:, :-1].T).T
    x = b_x[:, -1]

    return b, x, stable


def init_dyn(pm):
    """ Depending on the choice of degree approach and continuation variable
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

    if pm.c_var in ['kappa', 'eta_0', 'delta']:
        def dyn(b, x):
            setattr(locals()['pm'], pm.c_var, x)
            Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
            args = (0, b, pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                    pm.kappa, pm.k_mean)
            if pm.c_pmap:
                return poi_map(*args)-b
            else:
                return dyn_equ(*args)

    if pm.c_var in ['rho', 'r']:
        if pm.degree_approach == 'full':
            def dyn(b, x):
                setattr(locals()['pm'], pm.c_var, x)
                pm.a = pm.a_func(pm.r)
                Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
                args = (0, b, pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                        pm.kappa, pm.k_mean)
                if pm.c_pmap:
                    return poi_map(*args)-b
                else:
                    return dyn_equ(*args)

        elif pm.degree_approach == 'virtual':
            def dyn(b, x):
                setattr(locals()['pm'], pm.c_var, x)
                pm.a_v = pm.a_v_func(pm.r)
                Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
                args = (0, b, pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                        pm.kappa, pm.k_mean)
                if pm.c_pmap:
                    return poi_map(*args)-b
                else:
                    return dyn_equ(*args)

        elif pm.degree_approach == 'transform':
            def dyn(b, x):
                setattr(locals()['pm'], pm.c_var, x)
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


def null_vector(A):
    """ Compute normalised null vector to a given matrix A.
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


def partial_b(f, b_x, dh=1e-6):
    """ Compute partial derivative of f with respect to b.
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


def partial_x(f, b_x, dh=1e-6):
    """ Compute partial derivative of f with respect to variable x.
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


def jacobian(f, b_x):
    """ Compute Jacobi matrix containing the derivatives of f with respect to
    all states and the continuation variable x.

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
        [[df1_db1, df1_db2, ..., df1_dbn, df1_dx],
         [df2_db1, df2_db2, ..., df2_dbn, df2_dx],
          ...      ...      ...  ...      ...
         [dfn_db1, dfn_db2, ..., dfn_dbn, dfn_dx]]
    """

    df_db = partial_b(f, b_x)
    df_dx = partial_x(f, b_x)
    j = np.append(df_db, df_dx[:, None], 1)  # add df_dx as a last column

    return j


def newton_step(j, null, db_x, x_constrain):
    """ Stepping down a gradient in Newton-method fashion and enforcing a
    constrain on x.

    Parameters
    ----------
    j : ndarray, 2D float
        Jacobi matrix.
    null : ndarray, 1D float
        Null vector.
    db_x : ndarray, 1D float
        Dynamical direction (dyn(b_x)).
    x_constrain : float
        Constrain on x to ensure the solution stays on plane orthogonal to
        null vector and distance ds.

    Returns
    -------
    n_step : ndarray, 1D float
        Newton step.
    """

    m = np.append(j, null[None, :], 0)  # add null as last row
    n_step = np.linalg.solve(m, np.append(db_x, x_constrain))

    return n_step


def real_stack(x):
    """ Split a complex variable x into two parts and stack it to have a twice
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
    """ Reverse process of real_stack. Add second half of variable x as
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
    """ Compute an approximate basis for the nullspace of A.

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


def pmap_period(b, x, pm):
    """ Compute period times of poincare maps.

    Parameters
    ----------
    b : ndarray, 2D complex
        Curve of fixed points. [Curve, States]
    x : ndarray, 1D float
    pm : parameter.py
        Parameter file.

    Returns
    -------
    periods : ndarray, 1D float
        Times of how long poincare maps had to integrate.
    """

    periods = np.zeros(x.shape)
    for i in range(x.shape[0]):
        setattr(locals()['pm'], pm.c_var, x[i])

        if pm.degree_approach == 'full':
            pm.a = tn.generate.a_func_linear(pm.k_in, pm.k_out, pm.P_k,
                                             pm.N, pm.c,
                                             pm.i_prop, pm.j_prop)
            Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
            args = (0, b[i], pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                    pm.kappa, pm.k_mean)

        elif pm.degree_approach == 'virtual':
            pm.a_v = tn.generate.a_func_linear(pm.k_v_in, pm.k_v_out, pm.w,
                                               pm.N, pm.c,
                                               pm.i_prop, pm.j_prop)
            Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
            args = (0, b[i], pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                    pm.kappa, pm.k_mean)

        elif pm.degree_approach == 'transform':
            e = tn.utils.essential_fit(*pm.e_list, pm.r_list, pm.r)
            pm.usv = tn.utils.usv_from_essentials(*e, pm.c_in, pm.c_out)
            Q = tn.dynamics.degree_network.NQ_for_approach(pm)[1]
            args = (0, b[i], pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0, pm.delta,
                    pm.kappa, pm.k_mean)

        periods[i] = tn.dynamics.degree_network.\
            poincare_map(*args, return_time=True)[1]

    return periods
