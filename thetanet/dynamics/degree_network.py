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
    y : 1D array-like float
        State variable (theta).
    Gamma: 1D array-like float
        Factors arising from sophisticated double sum when rewriting the pulse
        function in sinusoidal form.
    n : int
        Sharpness parameter.
    d_n : float
        Normalisation from pulse function.
    Q : 2D array-like float
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
    dy/dt : 1D array-like float
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


def integrate(params, init=None):
    """ Perform time integration of mean field variables of neuronal network.

    Parameters
    ----------
    params : parameter.py
        Parameter file
    init : 1D array-like float
        Initial conditions

    Returns
    -------
    b_t : 2D array-like float [time, degree]
        Degree states at respective times.
    """

    if params.degree_approach == 'full':
        N_state_variables = params.N_k_in
        Q = np.squeeze(np.tensordot(params.P_k[..., None], params.a,
                                    axes=(1, 1)))

    elif params.degree_approach == 'virtual':
        N_state_variables = params.N_mu_in
        Q = np.squeeze(np.tesordot(params.w[..., None] * params.a_virt,
                                   axes=(1, 1)))

    elif params.degree_approach == 'trans':
        N_state_variables = params.N_k_in_occurrence
        Q = 1 / params.k_in_mean * params.E

    # Initialise network for t=0
    b_t = 1j * np.zeros((params.t_steps + 1, N_state_variables))
    if not init:
        init = 1j * np.zeros(N_state_variables)
    b_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('zvode')
    network.set_initial_value(b_t[0], params.t_start)
    network.set_f_params(params.Gamma, params.n, params.d_n, Q, params.eta_0,
                         params.delta, params.kappa)

    # Time integration
    print('Network with', N_state_variables, 'degrees | Integrating',
          params.t_end, 'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step <= params.t_steps:
        network.integrate(network.t + params.dt)
        b_t[step] = network.y
        progress = step / params.t_steps
        tn.utils.progress_bar(progress, time() - computation_start)
        step += 1
    print('    Successful. \n')

    return b_t
