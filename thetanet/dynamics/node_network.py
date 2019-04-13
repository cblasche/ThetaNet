import numpy as np
import thetanet as tn
import scipy.integrate
from scipy import sparse
from time import time


def dynamical_equation(t, y, d_n, n, kappa, k_in_mean, A, eta):
    """ Theta neuron model by Ermentrout and Kopell.

    Parameters
    ----------
    t : float
        Time
    y : ndarray, 1D float
        State variable (theta)
    d_n : float
        Normalisation from pulse function
    n : int
        Sharpness parameter
    kappa : float
        Coupling constant
    k_in_mean : float
        Mean value of all in-degrees
    A : ndarray, 2D int
        Adjacency matrix
    eta: ndarray, 1D float
        Intrinsic excitabilities

    Returns
    -------
    dy/dt : ndarray, 1D float
        Time derivative at time t.
    """

    P = d_n * (1-np.cos(y))**n
    I = A.dot(P) / k_in_mean
    dydt = 1-np.cos(y) + (1+np.cos(y))*(eta+kappa*I)

    return dydt


def integrate(pm, init=None):
    """ Perform time integration of neuronal network.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : ndarray, 1D float
        Initial conditions.

    Returns
    -------
    theta_t : ndarray, 2D float [time, neuron]
        Neuron states at respective times.
    """

    # Initialise network for t=0
    theta_t = np.zeros((pm.t_steps + 1, pm.N))
    # If init is not specified, choose uniform distribution.
    if init is None:
        init = np.linspace(0, 2 * np.pi * (1 - 1 / float(pm.N)), pm.N)
    theta_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('dopri5')
    network.set_initial_value(theta_t[0], pm.t_start)
    if pm.A.sum() / pm.N ** 2 < 0.4:  # Check for sparsity of A
        network.set_f_params(pm.d_n, pm.n, pm.kappa, pm.k_in_mean,
                             sparse.csc_matrix(pm.A.astype('float')), pm.eta)
    else:
        network.set_f_params(pm.d_n, pm.n, pm.kappa, pm.k_in_mean,
                             pm.A.astype('float'), pm.eta)

    # Time integration
    print('Network with', pm.N, 'nodes | Integrating', pm.t_end,
          'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step <= pm.t_steps:
        network.integrate(network.t + pm.dt)
        theta_t[step] = network.y
        progress = step / pm.t_steps
        tn.utils.progress_bar(progress, time()-computation_start)
        step += 1
    print('    Successful. \n')

    return theta_t
