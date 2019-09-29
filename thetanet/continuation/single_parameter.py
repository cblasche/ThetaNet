import thetanet as tn
from thetanet.exceptions import *
from thetanet.continuation import *
import time


def single_param(pm, init_b=None, init_x=None, init_stability=None,
                 adaptive_step_size=True, dh=1e-8):
    """
    Pseudo arc-length continuation scheme.
    Trace out a curve stable or unstable fixed points by continuing an initial
    solution under variation of the chosen variable pm.c_var.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init_b : ndarray, 2D complex
        Initial conditions for states followed by continuation variable.
    init_x : float
        Initial condition for continuation parameter.
    init_stability: bool
        True if fixed point is stable, False if not.
    adaptive_step_size: bool
        Adjust step size pm.c_ds according to speed of convergence if True.
    dh : float
        Precision when building difference quotient.

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

    # Minimal step size when algorithm should stop
    ds_min = 0.01 * abs(pm.c_ds)
    # Maximal step size which should not be exceeded
    ds_max = 100 * abs(pm.c_ds)

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
    dyn = init_dyn_1(pm)

    # To get the scheme started compute an initial null vector - the direction
    # of the first step. This should be done such that the change in c_var is
    # of the same sign as c_ds.
    j = jacobian(dyn, b_x[0], dh=dh)
    null = null_vector(j)
    if np.sign(null[-1]) < 0:
        null *= -1

    print('Network with', N_state_variables, 'degrees | Single-Parameter-'
          'Continuation of', pm.c_steps, 'steps:')
    t_start = time.time()
    try:
        for i in range(1, pm.c_steps+1):
            do_newton = True
            while do_newton:

                # Step forward along null vector
                b_x[i] = b_x[i - 1] + null * pm.c_ds

                # Converge back to stability with Newton scheme
                for n_i in range(pm.c_n):
                    try:
                        j = jacobian(dyn, b_x[i], dh=dh)
                    except NoPeriodicOrbitException:
                        raise NoPeriodicOrbitException(i)
                    g = real_stack(dyn(comp_unit(b_x[i, :-1]), b_x[i, -1]))
                    # Constrain on x to ensure the solution stays on plane
                    # orthogonal to null vector and distance ds.
                    x_constrain = (b_x[i] - b_x[i-1]).dot(null) - pm.c_ds
                    n_step = newton_step(j, null, g, x_constrain)

                    b_x[i] = b_x[i] - n_step

                    # if Newton's method is exact enough break
                    if np.abs(n_step).sum() < 1e-5:
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
                        if i is 1 and not pm.c_pmap:
                            print("\nCheck if there is a periodic orbit and if so"
                                  " use poincare map. (c_pmap=True)")
                            exit(1)
                        print("\nStep", i, "did not converge using minimal stepsize"
                                           " of " + str(ds_min) + "!")
                        raise ConvergenceError(i)
                else:
                    if do_newton:
                        print("\nStep", i, "did not converge!")
                        raise ConvergenceError(i)

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


def jacobian(f, b_x, dh=1e-6):
    """
    Compute Jacobi matrix containing the derivatives of f with respect to
    all states and the continuation variable x.

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
    j : ndarray, 2D float
        Jacobi matrix.(b is a real_stack)
        [[df1_db1, df1_db2, ..., df1_dbn, df1_dx],
         [df2_db1, df2_db2, ..., df2_dbn, df2_dx],
          ...      ...      ...  ...      ...
         [dfn_db1, dfn_db2, ..., dfn_dbn, dfn_dx]]
    """

    df_db = f_partial_b_1(f, b_x, dh=dh)
    df_dx = f_partial_x_1(f, b_x, dh=dh)
    j = np.append(df_db, df_dx[:, None], 1)  # add df_dx as a last column

    return j


def pmap_period(b, x, pm):
    """
    Compute period times of poincare maps.

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
