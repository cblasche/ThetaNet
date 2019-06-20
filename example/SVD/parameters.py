import numpy as np
import thetanet as tn
from scipy.stats import cauchy


""" Degree space and probability
"""
N = 5000  # number of neurons

k_in_min = 750  # lowest occurring node degree
k_in_max = 2000  # highest occurring node degree
k_in = np.arange(k_in_min, k_in_max + 1)
N_k_in = len(k_in)
P_k_in = k_in.astype(float) ** (-3)
P_k_in = P_k_in / np.sum(P_k_in)  # to have sum(P_k)=1
k_in_mean = np.sum(k_in * P_k_in)  # average value of node degrees

k_out_min = k_in_min
k_out_max = k_in_max
k_out = np.arange(k_out_min, k_out_max + 1)
N_k_out = len(k_out)
P_k_out = k_out.astype(float) ** (-3)
P_k_out = P_k_out / np.sum(P_k_out)  # to have sum(P_k)=1
k_out_mean = np.sum(k_out * P_k_out)

k = np.outer(k_in, k_out)
if k_in_mean == k_out_mean:
    k_mean = k_in_mean

P_k_func = tn.utils.bivar_pdf_rho(k_in, P_k_in, k_out, P_k_out)
rho = .0  # node correlation
P_k = P_k_func(rho)


""" Assortativity
"""
c = 0  # assortativity parameter
r = 0.0  # assortativity coefficient
i_prop = 'in'  # post-synaptic neuron property - 'in' / 'out'
j_prop = 'in'  # pre-synaptic neuron property


""" Node network
"""
K_in, K_out = tn.generate.degree_sequence_copula(P_k, N, k_in, k_out)
# A = tn.generate.configuration_model(K_in, K_out, r, i_prop, j_prop, simple=False)
A = tn.generate.chung_lu_model(K_in, K_out, c, i_prop, j_prop)


""" Degree network
"""
# There are different approaches to translate the neuronal network in its
# mean field description:
# 0: 'full'     - Compute the state for each degree of the network using an
#               designed assortativity function.
# 1: 'virtual'  - See 'full' but for a reduced set of degrees.
# 2: 'transform'- Use Carlo's method to transform the adjacency matrix into
#               degree space.
degree_approach_lib = ['full', 'virtual', 'transform']
degree_approach = degree_approach_lib[2]

# a_func = tn.generate.a_func_linear_r(k_in, k_out, P_k, N, i_prop, j_prop)
# a = a_func(r)


""" Virtual degree network
"""
# The degree space can be reduced due to the smooth nature of the state function
# to a smaller 'virtual' degree space. To be able to use N_mu virtual degrees
# one needs N_mu+1 polynomials q. The degree probability P_k transforms into
# N_mu weights w for the virtual degrees.
N_mu_in = 15
k_v_in, w_in, q_in = tn.utils.three_term_recursion(N_mu_in, k_in, P_k_in)
q_v_in = tn.utils.func_v(q_in, k_in, P_k_in)

N_mu_out = N_mu_in
k_v_out, w_out, q_out = tn.utils.three_term_recursion(N_mu_out, k_out, P_k_out)
q_v_out = tn.utils.func_v(q_out, k_out, P_k_out)

w_func = tn.utils.bivar_pdf_rho(k_v_in, w_in, k_v_out, w_out)
w = w_func(rho)

a_v = None
a_v_func = tn.generate.a_func_linear_r(k_v_in, k_v_out, w, N, i_prop, j_prop)
a_v = a_v_func(r)



""" Transform degree network
"""
N_c_in = 10
N_c_out = N_c_in
deg_k = 3  # degree of polynomial fit to basis functions u and v
m = 3  # svd rank
mapping = 'cumsum'  # 'cumsum' or 'linear' binning in k-space
E, B, c_in, c_out = tn.generate.a_func_transform(A, N_c_in, N_c_out, mapping=mapping)


""" Neuron dynamics
"""
kappa = -4  # influence strength of other pulses
eta_0 = 2.6  # center of distribution
delta = 0.1  # width of distribution
eta = cauchy.rvs(eta_0, delta, size=N)  # Cauchy- aka Lorentz-distribution
eta = np.clip(eta, eta_0 - 10, eta_0 + 10)
n = 2  # sharpness parameter of the pulses
d_n = 2 ** n * (np.math.factorial(n)) ** 2 \
      / float(np.math.factorial(2 * n))  # coefficient to normalise pulse
Gamma = tn.dynamics.degree_network.Gamma(n)     # coefficients for synaptic
                                                # current of mean field


""" Time
"""
t_start = 0
t_end = 200
t_steps = 2000  # write out steps from evolution (not related to precision!)
t = np.linspace(t_start, t_end, t_steps + 1)
dt = (t_end - t_start) * 1.0 / t_steps


""" Continuation
"""
c_lib = ['kappa', 'eta_0', 'delta', 'rho', 'r']
c_var = c_lib[4]
c_ds = 0.5  # step size
c_steps = 40  # number of steps
c_n = 7  # number of Newton iterations
c_pmap = False  # using poincare map of dynamical equation
