import numpy as np
import thetanet as tn

class DegreeSpace:
    def __init__(self, k_in, P_k_in, k_out, P_k_out):

        if not tn.generate.degree_sequences_match(k_in, P_k_in, k_out, P_k_out):
            raise ValueError('Change degree distributions to fulfill:'
                             ' k_in_mean = k_out_mean')

        self.i = k_in
        self.N_i = len(k_in)
        self.P_i = P_k_in
        self.o = k_out
        self.P_o = P_k_out
        self.N_o = len(k_out)
        self.mean = np.dot(k_in, P_k_in)


class DegreeSequence:
    def __init__(self, k, N, rho=0):
        self.i, self.o = tn.generate.degree_sequence(k, N, rho)
        self.mean = k.mean
