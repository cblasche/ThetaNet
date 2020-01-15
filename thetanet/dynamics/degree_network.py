import numpy as np
import thetanet as tn
import scipy.integrate
import scipy.sparse
from time import time
from thetanet.exceptions import *

""" The in- and output of dynamical_equation and poincare_map use state
    variables in flat form. It is possible to pass on an array of flattened
    states.
"""


def dynamical_equation(t, y, Gamma, n, d_n, Q, eta_0, delta, kappa, k_mean):
    """ Mean field description of the Theta neuron model.

    Parameters
    ----------
    t : float
        Time.
    y : ndarray, 1D float
        State variable. Length N_k_in * N_k_out in a long 1D vector.
    Gamma: array_like, 1D float
        Factors arising from sophisticated double sum when rewriting the pulse
        function in sinusoidal form.
    n : int
        Sharpness parameter.
    d_n : float
        Normalisation from pulse function.
    Q : ndarray, 2D float
        Connectivity matrix determining in-degree/in-degree connections.
    eta_0 : float
        Center of Cauchy distribution of intrinsic excitabilities.
    delta : float
        Half width at half maximum of Cauchy distribution of intrinsic
        excitabilities.
    kappa : float
        Coupling constant.
    k_mean : float
        Mean degree.

    Returns
    -------
    dy/dt : ndarray, 1D float
        Time derivative at time t.
    """

    P = Gamma[0]
    for p in range(1, n + 1):
        P += Gamma[p] * (y ** p + np.conjugate(y) ** p)
    P *= d_n
    if isinstance(Q, tuple):
        I = Q[0].T.dot((Q[1][:, None] * Q[2]).dot(P)) / k_mean  # u, s, v = Q
    else:
        I = Q.dot(P) / k_mean
    dydt = -1j * 0.5 * (y - 1) ** 2 + \
           1j * 0.5 * (y + 1) ** 2 * (eta_0 + 1j * delta + kappa * I)

    return dydt


def integrate(pm, init=None, console_output=True):
    """ Perform time integration of mean field variables of neuronal network.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : ndarray, 2D complex
        Initial conditions.
    console_output: bool
        Whether or not to print details to the console.

    Returns
    -------
    b_t : ndarray, 3D float [time, degree(2D)]
        Degree states at respective times.
    """

    N_state_variables, Q = NQ_for_approach(pm)

    # Initialise network for t=0
    b_t = np.zeros((len(pm.t), N_state_variables)).astype(complex)

    if init is None:
        init = np.zeros(N_state_variables)

    try:
        init.shape = N_state_variables
    except ValueError:
        print('\n Invalid initial conditions! Check length of initial '
              'condition and chosen degree approach.')
        quit()

    b_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('zvode')
    network.set_initial_value(b_t[0], pm.t[0])
    network.set_f_params(pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0,
                         pm.delta, pm.kappa, pm.k_mean)

    # Time integration
    if console_output:
        print('\nNetwork with', N_state_variables, 'degrees | Integrating',
              pm.t[-1], 'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step < len(pm.t):
        network.integrate(pm.t[step])
        b_t[step] = network.y
        if console_output:
            progress = step / (len(pm.t))
            tn.utils.progress_bar(progress, time() - computation_start)
        step += 1

    return b_t


def Gamma(n):
    """ Parameters necessary for the mean field dynamics.

    Laing, C. R.; Derivation of a neural field model from a network of theta
    neurons; Physical Review E, 2014  -  equation (14)

    Parameters
    ----------
    n : int
        Sharpness parameter.

    Returns
    -------
    Gamma : ndarray, 1D float
        Array holding the parameters Gamma(n)
    """

    g = np.zeros(n+1)
    for p in range(n+1):
        for l in range(n+1):
            for m in range(l+1):
                g[p] += ((l-2*m) == p) * np.math.factorial(n)*(-1)**l / \
                    float(2**l * np.math.factorial(n-l) * np.math.factorial(m)
                          * np.math.factorial(l-m))

    return g


def NQ_for_approach(pm):
    """ In order to use the same dynamical equation the number of state
    variables and the transition matrix need to be computed according to
    the chosen degree_approach.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.

    Returns
    -------
    N_state_variables : int
        Number of state variables.
    Q : ndarray, 2D
        Transition matrix - a weighted assortativity function.
    """

    if pm.degree_approach == 'full':
        N_state_variables = pm.N_k_in * pm.N_k_out
        Q = pm.P_k.flatten()[None, :] * pm.N * \
            pm.a.reshape(N_state_variables, N_state_variables)

    if pm.degree_approach == 'full_in_only':
        N_state_variables = pm.N_k_in
        Q = (pm.P_k[None, ...] * pm.N * pm.a.mean(1)).sum(2)

    elif pm.degree_approach == 'virtual':
        N_state_variables = pm.N_mu_in * pm.N_mu_out
        Q = pm.w.flatten()[None, :] * pm.P_k_v.flatten()[None, :] * pm.N * \
            pm.a_v.reshape(N_state_variables, N_state_variables)

    elif pm.degree_approach == 'virtual_in_only':
        N_state_variables = pm.N_mu_in
        Q = (pm.w[None, ...] * pm.P_k_v[None, ...] * pm.N * pm.a_v.mean(1)).sum(2)

    elif pm.degree_approach == 'transform':
        N_state_variables = pm.N_c_in * pm.N_c_out
        Q = pm.usv

    else:
        print('\n "degree_approach" is none of the possible choices',
              '"full", "full_in_only", "virtual", "virtual_in_only",'
              ' "transform".')
        quit(1)

    return N_state_variables, Q


def poincare_map(t, y, Gamma, n, d_n, Q, eta_0, delta, kappa, k_mean,
                 return_time=False):
    """ Perform time integration to the next intersection with poincare section.

    Parameters
    ----------
    t : float
        Time.
    y : ndarray, 1D float
        State variable. (flattened)
    Gamma: array_like, 1D float
        Factors arising from sophisticated double sum when rewriting the pulse
        function in sinusoidal form.
    n : int
        Sharpness parameter.
    d_n : float
        Normalisation from pulse function.
    Q : ndarray, 2D float
        Connectivity matrix determining in-degree/in-degree connections.
    eta_0 : float
        Center of Cauchy distribution of intrinsic excitabilities.
    delta : float
        Half width at half maximum of Cauchy distribution of intrinsic
        excitabilities.
    kappa : float
        Coupling constant.
    k_mean : float
        Mean degree.
    return_time : bool
        Return elapsed time in addition to updated state variable.

    Returns
    -------
    y : ndarray, 2D float [time, degree]
        Degree states (flattened) at respective times.
    """

    t0 = np.copy(t)
    y0 = np.copy(y)

    def f(t, y):
        f = dynamical_equation(t, y, Gamma, n, d_n, Q, eta_0, delta, kappa,
                               k_mean)
        return f

    def cycle_closed(t, y):
        if t != t0:  # skip the first moment, when it is obviously 0
            return f(t, y)[0].real  # follow the limit cycle and let the
                                    # section be its minimum.
        else:
            return 1
    cycle_closed.terminal = True
    cycle_closed.direction = 1

    solver = scipy.integrate.solve_ivp(f, [t0, t0+100], y0, dense_output=True,
                                   events=cycle_closed, rtol=1e-7, atol=1e-7)
    t = np.asarray(solver.t_events).squeeze()

    if t:
        y = solver.sol(t)
    else:
        print("\nTrajectory not crossing Poincare section! Potentially not a "
              "periodic orbit.")
        raise NoPeriodicOrbitException

    if return_time:
        return y, t
    else:
        return y
