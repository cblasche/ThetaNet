import parameters as pm
import thetanet as tn
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

"""Example: SVD parameterisation

We create a matrix A and mix it for a range of (in-in)-assortativity values.
Transforming those adjacency matrices will lead to E-matrices which can be used
for the mean field description.
To these matrices we apply the SVD scheme and fit polynomials through basis 
vectors and singular values. Using the coefficients of these polynomials we can
construct an E-matrix with arbitrary assortativity.
"""

r_stack = np.linspace(0, 0.4, 5)

K_in, K_out = tn.generate.degree_sequence(pm.k_in, pm.P_k_in, pm.k_out, pm.P_k_out, pm.N, pm.rho)
A = tn.generate.configuration_model(K_in, K_out)

A_stack = np.empty((len(r_stack), pm.N, pm.N), dtype=int)
for i_r in range(len(r_stack)):
    tn.generate.assortative_mixing(A, r_stack[i_r], 'in', 'in')
    A_stack[i_r] = A
coef = tn.utils.svd_coef_E(A_stack, r_stack, 3, 5, pm.k_in, m=2)
# If coef needs to be saved, do it in the following way:
# np.savez('coef', u=coef[0], v=coef[1], s=coef[2])
# coef = np.load('coef.npz')
# coef = coef['u'], coef['v'], coef['s']

# For an E-matrix of the value r we write:
r = 0.25
Esvd = tn.utils.svd_E(coef, r)


# Let us compare the dynamics to simulation with r=0.2 and r=0.3.

plt.figure(0)
pm.A = A_stack[2]
pm.E, pm.B = tn.generate.a_func_transform(pm.A, pm.k_in)
b_t = tn.dynamics.degree_network.integrate(pm)[-30:]
R = np.tensordot(pm.B, b_t, axes=(1, 1)).mean(axis=0)
plt.scatter(R.real, R.imag, label='r=0.2')

pm.A = A_stack[3]
pm.E, pm.B = tn.generate.a_func_transform(pm.A, pm.k_in)
b_t = tn.dynamics.degree_network.integrate(pm)[-30:]
R = np.tensordot(pm.B, b_t, axes=(1, 1)).mean(axis=0)
plt.scatter(R.real, R.imag, label='r=0.3')

pm.E = Esvd  # pm.B is shared by all of them
b_t = tn.dynamics.degree_network.integrate(pm)[-30:]
R = np.tensordot(pm.B, b_t, axes=(1, 1)).mean(axis=0)
plt.scatter(R.real, R.imag, label='r=0.25')

plt.legend()
plt.show()
