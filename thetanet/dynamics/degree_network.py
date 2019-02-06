import numpy as np
import thetanet as tn
import scipy.integrate
import scipy.sparse
from time import time


def dynamical_equation(t, y, Gamma, n, d_n, Q, eta_0, delta, kappa):
    """ Mean field description of the Theta neuron model.

    Parameters
    ----------
    t : float
        Time.
    y : array_like, 1D float
        State variable (theta).
    Gamma: array_like, 1D float
        Factors arising from sophisticated double sum when rewriting the pulse
        function in sinusoidal form.
    n : int
        Sharpness parameter.
    d_n : float
        Normalisation from pulse function.
    Q : array_like, 2D float
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
    dy/dt : array_like, 1D float
        Time derivative at time t.
    """

    P = Gamma[0]
    for p in range(1, n + 1):
        P += Gamma[p] * (y ** p + np.conjugate(y) ** p)
    P *= d_n
    I = np.tensordot(Q, P, axes=(0, 0))  # axis 0 refers to k^\prime_in
    dydt = -1j * 0.5 * (y - 1) ** 2 + \
           1j * 0.5 * (y + 1) ** 2 * (eta_0 + 1j * delta + kappa * I)

    return dydt


def integrate(pm, init=None):
    """ Perform time integration of mean field variables of neuronal network.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : array_like, 1D float
        Initial conditions.

    Returns
    -------
    b_t : array_like, 2D float [time, degree]
        Degree states at respective times.
    """

    if pm.degree_approach == 'full':
        N_state_variables = pm.N_k_in
        Q = pm.P_k_in[:, None] * pm.a
        # Q = np.squeeze(np.tensordot(pm.P_k[..., None], pm.a,
        #                            axes=(1, 1)))

    elif pm.degree_approach == 'virtual':
        N_state_variables = pm.N_mu_in
        Q = pm.w_in[:, None] * pm.a_v

    elif pm.degree_approach == 'transform':
        N_state_variables = pm.N_k_in
        Q = 1 / pm.k_in_mean * pm.E

    else:
        print('\n "degree_approach" is none of the possible choices "full",'
              '"virtual" or "transform".')
        quit(1)

    # Initialise network for t=0
    b_t = 1j * np.zeros((pm.t_steps + 1, N_state_variables))
    if not init:
        init = 1j * np.zeros(N_state_variables)
    b_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('zvode')
    network.set_initial_value(b_t[0], pm.t_start)
    network.set_f_params(pm.Gamma, pm.n, pm.d_n, Q, pm.eta_0,
                         pm.delta, pm.kappa)

    # Time integration
    print('Network with', N_state_variables, 'degrees | Integrating',
          pm.t_end, 'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step <= pm.t_steps:
        network.integrate(network.t + pm.dt)
        b_t[step] = network.y
        progress = step / pm.t_steps
        tn.utils.progress_bar(progress, time() - computation_start)
        step += 1
    print('    Successful. \n')

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
    Gamma : array_like, 1D float
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
