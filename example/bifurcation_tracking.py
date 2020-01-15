import parameters as pm
import thetanet as tn
import numpy as np
from matplotlib import pyplot as plt


def main():
    # make usv_func in parameter.py available
    svd_list = np.load('svd_list.npz')
    pm.usv_func = tn.utils.usv_func_from_svd_list(svd_list)

    init_rho = -0.8
    pm.rho = init_rho
    pm.usv = pm.usv_func(pm.rho)

    b, x, stable = tn.continuation.single_param(pm)
    sn = (np.where(stable[1:] != stable[:-1])[0] + 1).tolist()

    def continue_and_plot(i):
        pm.c_ds = 0.05  # continue from init_rho to positive direction
        pm.c_steps = 130
        x_sn, y_sn = tn.continuation.saddle_node(pm, init_b=b[i], init_x=x[i], init_y=init_rho)[1:]
        plt.plot(x_sn, y_sn, c='k')

    [continue_and_plot(i) for i in sn]
    plt.axis([-0.7, -0.3, -0.8, 0.8])
    plt.xlabel(r'$\eta_0$')
    plt.ylabel(r'$\hat{\rho}$')
    plt.show()


if __name__ == '__main__':
    main()
