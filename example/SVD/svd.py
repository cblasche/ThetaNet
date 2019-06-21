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
vectors and singular values. Interpolating between those values we can
construct E-matrices with arbitrary assortativity.
"""

# B is shared by all E matrices
B, c_in, c_out = tn.generate.a_func_transform(pm.A, pm.N_c_in, pm.N_c_out)[1:]

r_list = np.linspace(0, 0.4, 5)

A_list = np.empty((len(r_list), pm.N, pm.N), dtype=int)
for i_r in range(len(r_list)):
    tn.generate.assortative_mixing(pm.A, r_list[i_r], 'in', 'in')
    A_list[i_r] = pm.A

e_list = tn.utils.essential_list_from_data(A_list, pm.N_c_in, pm.N_c_out, m=pm.m, deg_k=pm.deg_k)
# # If coef needs to be saved, do it in the following way:
# np.savez('e_list', u=e_list[0], s=e_list[1], v=e_list[2])
# e_list = np.load('e_list.npz')
# e_list = e_list['u'], e_list['s'], e_list['v']

# For the dynamics we require u,s,v:
r = 0.25
e = tn.utils.essential_fit(*e_list, r_list, r)
pm.usv = tn.utils.usv_from_essentials(*e, c_in, c_out)
# But E can also be computed:
E = tn.utils.E_from_usv(*pm.usv)

plt.figure(0)
b_t = tn.dynamics.degree_network.integrate(pm)[-1000:]
R = B.dot(b_t.T).mean(0)
plt.plot(R.real, R.imag, label='r=0.25')


# Let's compare the dynamics to simulations with r=0.2 and r=0.3.
pm.A = A_list[2]
E = tn.generate.a_func_transform(pm.A, pm.N_c_in, pm.N_c_out)[0]
pm.usv = tn.utils.usv_from_E(E, pm.m)
b_t = tn.dynamics.degree_network.integrate(pm)[-1000:]
R = B.dot(b_t.T).mean(0)
plt.plot(R.real, R.imag, label='r=0.2')

pm.A = A_list[3]
E = tn.generate.a_func_transform(pm.A, pm.N_c_in, pm.N_c_out)[0]
pm.usv = tn.utils.usv_from_E(E, pm.m)
b_t = tn.dynamics.degree_network.integrate(pm)[-1000:]
R = pm.B.dot(b_t.T).mean(0)
plt.plot(R.real, R.imag, label='r=0.3')

plt.legend()
plt.show()
