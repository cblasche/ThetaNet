import parameters as pm
import thetanet as tn
import numpy as np
from matplotlib import pyplot as plt

"""Example: Continuation

We trace out a curve in parameter space of (in-in)-assortativity. In the
following example parameters are chosen to find a periodic orbit and
accordingly c_pmap=True for using a Poincare map.
For a positive c_ds we will explore the space in positive direction and vice
versa.
"""

bx, x, stable = tn.utils.continuation(pm)
periods = tn.utils.pmap_period(bx, x, pm)
R = np.tensordot(pm.w, bx, axes=((0, 1), (1, 2)))

plt.figure(1)
plt.scatter(x, periods, c=stable)
plt.show()
