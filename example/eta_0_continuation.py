import parameters as pm
import thetanet as tn
import numpy as np
from matplotlib import pyplot as plt


def main():
    # make usv_func in parameter.py available
    svd_list = np.load('svd_list.npz')
    pm.usv_func = tn.utils.usv_func_from_svd_list(svd_list)
    pm.B = svd_list['B']

    def continue_and_plot(rho, c):
        # update values in the parameter file
        pm.rho = rho
        pm.usv = pm.usv_func(pm.rho)
        pm.ds = -0.05
        pm.eta_0 = 0

        # run continuation and compute average firing frequency
        b, x, stable = tn.continuation.single_param(pm)
        z = pm.B.dot(b.T).mean(0)
        f = 1 / np.pi * ((1 - z) / (1 + z)).real

        # slicing unstable and stable parts for plots
        sn = (np.where(stable[1:] != stable[:-1])[0] + 1).tolist()
        f = [f[i:j] for i, j in zip([0] + sn, sn + [None])]
        x = [x[i:j] for i, j in zip([0] + sn, sn + [None])]
        stable = [stable[i:j] for i, j in zip([0] + sn, sn + [None])]

        # plot
        ls_dict = {0.: '--', 1.: '-'}
        for ff, xx, ss in zip(f, x, stable):
            plt.plot(xx, ff, ls=ls_dict[ss[0]], c=c)

    # continuation for 3 rho values and plot them in assigned colors
    [continue_and_plot(rho, c) for rho, c in zip([-0.7, 0, 0.55], ['r', 'k', 'b'])]
    plt.axis([-0.7, -0.2, 0, 0.4])
    plt.xlabel(r'$\eta_0$')
    plt.ylabel('f')
    plt.show()


if __name__ == '__main__':
    main()
