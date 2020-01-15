import numpy as np
import thetanet as tn

""" Degree space and probability
"""
N = 2000  # number of neurons

k_in_min = 100  # lowest occurring node degree
k_in_max = 400  # highest occurring node degree
k_in = np.arange(k_in_min, k_in_max + 1)
N_k_in = len(k_in)

k_out = np.copy(k_in)
N_k_out = len(k_out)

P_k_in = k_in.astype(float) ** (-3)
P_k_in = P_k_in / np.sum(P_k_in)  # to have sum(P_k)=1
P_k_out = np.copy(P_k_in)

k_mean = np.sum(k_in * P_k_in)  # average value of node degrees


""" Degree network
"""
degree_approach = 'transform'
N_c_in = 10  # number of degree clusters
N_c_out = N_c_in


""" Neuron dynamics
"""
# coupling strength
kappa = 1.5

# pulse function
n = 2  # sharpness parameter
d_n = 2**n * (np.math.factorial(n)) ** 2 /\
      float(np.math.factorial(2*n))  # normalisation factor
Gamma = tn.dynamics.degree_network.Gamma(n)  # ensemble coefficients

# eta ’s drawn from Lorentzian ( Cauchy ) distribution
eta_0 = .0  # center of distribution
delta = 0.05  # width of distribution

# time
t = np.linspace(0, 30, 1000)


""" Continuation
"""
c_var = 'eta_0'  # parameter for single parameter continuation
c_var2 = 'rho'  # additional parameter for bifurcation tracking
c_ds = -0.05  # step size
c_steps = 80  # number of steps
c_n = 7  # number of Newton iterations
c_pmap = False  # continue poincare-map?