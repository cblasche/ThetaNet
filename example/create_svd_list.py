import parameters as pm
import thetanet as tn
import numpy as np


def main():
    def A_of_rho(rho):
        P_k = tn.utils.bivar_pdf(pm.P_k_in, pm.P_k_out, rho_gauss=rho)
        K_in, K_out = tn.generate.degree_sequence_copula(P_k, pm.N, pm.k_in, pm.k_out,
                                                         console_output=False)
        A = tn.generate.chung_lu_model(K_in, K_out)
        return A

    rho_list = np.linspace(-0.99, 0.99, 6)
    A_list = np.asarray([A_of_rho(rho) for rho in rho_list])

    e_list = tn.utils.essential_list_from_data(A_list, pm.N_c_in, pm.N_c_out)
    B, c_in, c_out = tn.generate.a_func_transform(A_list[0], pm.N_c_in, pm.N_c_out)[1:]

    np.savez('svd_list', u=e_list[0], s=e_list[1], v=e_list[2], r_list=rho_list,
             B=B, c_in=c_in, c_out=c_out)


if __name__ == '__main__':
    main()