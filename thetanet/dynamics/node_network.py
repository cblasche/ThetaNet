import numpy as np
import thetanet as tn
import scipy.integrate
from scipy import sparse
from time import time


def dynamical_equation(t, y, d_n, n, kappa, k_mean, A, eta):
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
    k_mean: float
        Mean degree.
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
    I = A.dot(P) / k_mean
    dydt = 1-np.cos(y) + (1+np.cos(y))*(eta+kappa*I)

    return dydt


def integrate(pm, init=None, console_output=True):
    """ Perform time integration of neuronal network.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : ndarray, 1D float
        Initial conditions.
    console_output: bool
        Whether or not to print details to the console.

    Returns
    -------
    theta_t : ndarray, 2D float [time, neuron]
        Neuron states at respective times.
    """

    # Initialise network for t=0
    theta_t = np.zeros((len(pm.t), pm.N))
    # If init is not specified, choose uniform distribution.
    if init is None:
        init = np.linspace(0, 2 * np.pi * (1 - 1 / float(pm.N)), pm.N)
    theta_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('dopri5')
    network.set_initial_value(theta_t[0], pm.t[0])
    if pm.A.sum() / pm.N ** 2 < 0.4:  # Check for sparsity of A
        network.set_f_params(pm.d_n, pm.n, pm.kappa, pm.k_mean,
                             sparse.csc_matrix(pm.A.astype('float')), pm.eta)
    else:
        network.set_f_params(pm.d_n, pm.n, pm.kappa, pm.k_mean,
                             pm.A.astype('float'), pm.eta)

    # Time integration
    if console_output:
        print('\nNetwork with', pm.N, 'nodes | Integrating', pm.t[-1],
          'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step <= len(pm.t):
        network.integrate(network.t + (pm.t[1] - pm.t[0]))
        theta_t[step] = network.y
        if console_output:
            progress = step / (len(pm.t))
            tn.utils.progress_bar(progress, time()-computation_start)
        step += 1

    return theta_t


def dynamical_equation_mean(t, y, Gamma, n, d_n, A, eta_0, delta, kappa,
                            k_mean):
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
    A : ndarray, 2D float
        Adjacency matrix.
    eta_0 : float
        Center of Cauchy distribution of intrinsic excitabilities.
    delta : float
        Half width at half maximum of Cauchy distribution of intrinsic
        excitabilities.
    kappa : float
        Coupling constant.
    k_mean: float
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
    I = A.dot(P) / k_mean
    dydt = -1j * 0.5 * (y - 1) ** 2 + \
           1j * 0.5 * (y + 1) ** 2 * (eta_0 + 1j * delta + kappa * I)

    return dydt


def integrate_mean(pm, init=None, console_output=True):
    """ Perform time integration of neuronal network using Ott/Antonsen.
    Ensemble of different realisations of random eta draws.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : ndarray, 1D float
        Initial conditions.
    console_output: bool
        Whether or not to print details to the console.

    Returns
    -------
    z_t : ndarray, 2D float [time, neuron]
        Mean value <e^(i*theta)> at respective times.
    """

    # Initialise network for t=0
    z_t = np.zeros((len(pm.t), pm.N)).astype(complex)
    # If init is not specified, choose uniform distribution.
    if init is None:
        init = np.zeros(pm.N)
    z_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation_mean)
    network.set_integrator('zvode')
    network.set_initial_value(z_t[0], pm.t[0])
    if pm.A.sum() / pm.N ** 2 < 0.4:  # Check for sparsity of A
        network.set_f_params(pm.Gamma, pm.n, pm.d_n,
                             sparse.csc_matrix(pm.A.astype('float')), pm.eta_0,
                             pm.delta, pm.kappa, pm.k_mean)
    else:
        network.set_f_params(pm.Gamma, pm.n, pm.d_n, pm.A.astype('float'),
                             pm.eta_0, pm.delta, pm.kappa, pm.k_mean)

    # Time integration
    if console_output:
        print('\nNetwork with', pm.N, 'nodes | Integrating mean field',
              pm.t[-1], 'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step < len(pm.t):
        # network.integrate(network.t + (pm.t[1] - pm.t[0]))
        network.integrate(pm.t[step])
        z_t[step] = network.y
        if console_output:
            progress = step / (len(pm.t))
            tn.utils.progress_bar(progress, time()-computation_start)
        step += 1

    return z_t
