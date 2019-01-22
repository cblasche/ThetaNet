import numpy as np
import thetanet as tn
import scipy.integrate
import scipy.sparse
from time import time


def dynamical_equation(t, y, d_n, n, kappa, k_in_mean, A, eta):
    """ Theta neuron model by Ermentrout and Kopell.

    Parameters
    ----------
    t : float
        Time
    y : 1D array-like float
        State variable (theta)
    d_n : float
        Normalisation from pulse function
    n : int
        Sharpness parameter
    kappa : float
        Coupling constant
    k_in_mean : float
        Mean value of all in-degrees
    A : 2D array-like int
        Adjacency matrix
    eta: 1D array-like float
        Intrinsic excitabilities

    Returns
    -------
    dy/dt : 1D array-like float
        Time derivative at time t.
    """

    P = d_n * (1-np.cos(y))**n
    I = np.dot(A, P) / k_in_mean
    dydt = 1-np.cos(y) + (1+np.cos(y))*(eta+kappa*I)

    return dydt


def integrate(params, init=None):
    """ Perform time integration of neuronal network.

    Parameters
    ----------
    params : parameter.py
        Parameter file
    init : 1D array-like float
        Initial conditions

    Returns
    -------
    theta_t : 2D array-like float [time, neuron]
        Neuron states at respective times.
    """

    # Initialise network for t=0
    theta_t = np.zeros((params.t_steps + 1, params.N))
    # If init is not specified, choose uniform distribution.
    if not init:
        init = np.linspace(0, 2 * np.pi * (1 - 1 / float(params.N)), params.N)
    theta_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('dopri5')
    network.set_initial_value(theta_t[0], params.t_start)
    network.set_f_params(params.d_n, params.n, params.kappa, params.k_in_mean,
                         params.A.astype('float'), params.eta)

    # Time integration
    print('Network with', params.N, 'nodes | Integrating', params.t_end,
          'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step <= params.t_steps:
        network.integrate(network.t + params.dt)
        theta_t[step] = network.y
        progress = step / params.t_steps
        tn.utils.progress_bar(progress, time()-computation_start)
        step += 1
    print('    Successful. \n')

    return theta_t