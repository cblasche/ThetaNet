import numpy as np
import thetanet as tn
import scipy.integrate
import scipy.sparse
from time import time
from thetanet.exceptions import *

""" The in- and output of dynamical_equation and poincare_map use state
    variables in flat form. It is possible to pass on an array of the flat
    states.
"""


def dynamical_equation(t, y, Gamma, n, d_n, Q, eta_0, delta, kappa):
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

    Returns
    -------
    dy/dt : ndarray, 1D float
        Time derivative at time t.
    """
    y_shape = y.shape
    y.shape = (Q.shape[0], Q.shape[1]) + y.shape[1:]

    P = Gamma[0]
    for p in range(1, n + 1):
        P += Gamma[p] * (y ** p + np.conjugate(y) ** p)
    P *= d_n
    I = np.tensordot(Q, P, axes=((2, 3), (0, 1)))  # axis 2/3 refers to k^\prime
    dydt = -1j * 0.5 * (y - 1) ** 2 + \
           1j * 0.5 * (y + 1) ** 2 * (eta_0 + 1j * delta + kappa * I)

    y.shape = y_shape
    dydt.shape = y_shape
    return dydt


def integrate(pm, init=None):
    """ Perform time integration of mean field variables of neuronal network.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : ndarray, 2D complex
        Initial conditions.

    Returns
    -------
    b_t : ndarray, 3D float [time, degree(2D)]
        Degree states at respective times.
    """

    N_state_variables, Q = NQ_for_approach(pm)

    # Initialise network for t=0
    b_t = np.zeros((pm.t_steps + 1, N_state_variables)).astype(complex)

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
    network.set_initial_value(b_t[0], pm.t_start)
    network.set_f_params(pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0,
                         pm.delta, pm.kappa)

    # Time integration
    print('\nNetwork with', N_state_variables, 'degrees | Integrating',
          pm.t_end, 'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step <= pm.t_steps:
        network.integrate(network.t + pm.dt)
        b_t[step] = network.y
        progress = step / pm.t_steps
        tn.utils.progress_bar(progress, time() - computation_start)
        step += 1

    b_t.shape = (b_t.shape[0], Q.shape[0], Q.shape[1])
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
    Q : ndarray, 4D
        Transition matrix - a weighted assortativity function.
    """

    if pm.degree_approach == 'full':
        N_state_variables = pm.N_k_in * pm.N_k_out
        Q = pm.P_k[None, None, :, :] * pm.a * (pm.N / pm.k_mean)

    elif pm.degree_approach == 'virtual':
        N_state_variables = pm.N_mu_in * pm.N_mu_out
        Q = pm.w[None, None, :, :] * pm.a_v * (pm.N / pm.k_mean)

    elif pm.degree_approach == 'transform':
        N_state_variables = pm.N_k_in
        Q = 1 / pm.k_in_mean * pm.E

    else:
        print('\n "degree_approach" is none of the possible choices',
              pm.degree_approach_lib, '.')
        quit(1)

    return N_state_variables, Q


def poincare_map(t, y, Gamma, n, d_n, Q, eta_0, delta, kappa,
                 return_time=False):
    """ Perform time integration to the next intersection with poincare section.

    Parameters
    ----------
    t : float
        Time.
    y : ndarray, 1D float
        State variable. (flattend)
    Gamma: array_like, 1D float
        Factors arising from sophisticated double sum when rewriting the pulse
        function in sinusoidal form.
    n : int
        Sharpness parameter.
    d_n : float
        Normalisation from pulse function.
    Q : ndarray, 4D float
        Connectivity matrix determining in-degree/in-degree connections.
    eta_0 : float
        Center of Cauchy distribution of intrinsic excitabilities.
    delta : float
        Half width at half maximum of Cauchy distribution of intrinsic
        excitabilities.
    kappa : float
        Coupling constant.
    return_time : bool
        Return elapsed time in addition to updated state variable.

    Returns
    -------
    y : ndarray, 2D float [time, degree]
        Degree states (flattend) at respective times.
    """

    t0 = np.copy(t)
    y0 = np.copy(y)

    def f(t, y):
        f = tn.dynamics.degree_network.dynamical_equation(t, y, Gamma, n, d_n,
                                                          Q, eta_0, delta,
                                                          kappa)
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
