import numpy as np
import thetanet as tn


def correlation_from_matrix(A):
    """ Compute Pearson correlation coefficient for an adjacency matrix.

    Parameters
    ----------
    A : ndarray, 2D int
        Adjacency matrix.

    Returns
    -------
    rho : float
        Pearson correlation coefficient.
    """

    K_in = A.sum(1)
    K_out = A.sum(0)

    rho = tn.generate.correlation_from_sequences(K_in, K_out)

    return rho


def chung_lu_model(K_in, K_out, k_in=None, k_out=None, P_k=None, c=0,
                   i_prop='in', j_prop='out'):
    """ Chung Lu style adjacency matrix creation. Generating a target matrix T
    and compare it with a matrix filled with random variables. Assortativity
    may be induced by a non-zero c value.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    k_in : ndarray, 1D int
        In-degree space.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k : ndarray, 1D float
        Joint In-Out-degree probability.
    c : float
        Assortativity parameter.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    A : ndarray, 2D int
        Adjacency matrix.
    """

    N = K_in.shape[0]

    T = np.outer(K_in, K_out) / (N * K_in.mean())

    if c !=0:
        P_k_in = P_k.sum(1)
        P_k_out = P_k.sum(0)
        k_mean = P_k_in.dot(k_in)

        k_mean_edge_i_in = P_k_in.dot(k_in ** 2) / k_mean
        k_mean_edge_i_out = (P_k * np.outer(k_in, k_out)).sum() / k_mean
        k_mean_edge_j_in = k_mean_edge_i_out
        k_mean_edge_j_out = P_k_out.dot(k_out ** 2) / k_mean

        k_mean_edge_i = eval('k_mean_edge_i_' + i_prop)
        k_mean_edge_j = eval('k_mean_edge_j_' + j_prop)
        K_i = eval('K_' + i_prop)
        K_j = eval('K_' + j_prop)

        T += c * np.outer((K_i - k_mean_edge_i), (K_j - k_mean_edge_j)) \
            / (N * K_in.mean())

    A = np.random.uniform(size=(N, N)) < T

    return A.astype(int)


def assortative_mixing_float(T, r, i_prop='in', j_prop='out'):
    """ Change the target adjacency matrix such that the realisation of
    a matrix with desired assortativity coefficient is more likely.

    Since all neurons have a probability to be connected, pick a quadruple of
    neurons where two of them are pre- and two are post-synaptic neurons.
    Redistribute their probability by moving some probability from
    one connection to the other.

    Parameters
    ----------
    T : ndarray, 2D float
        Target adjacency matrix.
    r : float
        Assortativity coefficient.
    i_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').
    j_prop : str
        Respective node degree which is involved in assortative mixing.
        ('in' or 'out').

    Returns
    -------
    T will be modified.
    """

    K_in = T.sum(axis=1)
    K_out = T.sum(axis=0)
    K_i_prop = eval('K_' + i_prop)
    K_j_prop = eval('K_' + j_prop)

    K_mean = K_in.mean()

    r_diff = r - tn.generate.a_coef_from_matrix(T, i_prop, j_prop)
    N = len(K_in)
    N_quadruple = int(N / 4)
    while abs(r_diff) > 0.01:
        pool = np.random.choice(N, N_quadruple * 4, replace=False)
        I_i, I_j, J_i, J_j = np.split(pool, 4)  # edge I goes from neuron
                                                # I_j to neuron I_i
        T_max = np.max([T[I_i, I_j], T[I_i, J_j], T[J_i, I_j], T[J_i, J_j]],
                       axis=0)  # maximum probability
        T_min = np.min([T[I_i, I_j], T[I_i, J_j], T[J_i, I_j], T[J_i, J_j]],
                       axis=0)  # minimum probability
        ex_val = np.min([1 - T_max, T_min])*0.8  # exchanged probability
        cor_plus = (T[I_i, I_j] + ex_val) * (K_i_prop[I_i] - K_mean) * (K_j_prop[I_j] - K_mean) + \
                   (T[J_i, J_j] + ex_val) * (K_i_prop[J_i] - K_mean) * (K_j_prop[J_j] - K_mean) + \
                   (T[J_i, I_j] - ex_val) * (K_i_prop[J_i] - K_mean) * (K_j_prop[I_j] - K_mean) + \
                   (T[I_i, J_j] - ex_val) * (K_i_prop[I_i] - K_mean) * (K_j_prop[J_j] - K_mean)
        cor_minus = (T[I_i, I_j] - ex_val) * (K_i_prop[I_i] - K_mean) * (K_j_prop[I_j] - K_mean) +\
                   (T[J_i, J_j] - ex_val) * (K_i_prop[J_i] - K_mean) * (K_j_prop[J_j] - K_mean) + \
                   (T[J_i, I_j] + ex_val) * (K_i_prop[J_i] - K_mean) * (K_j_prop[I_j] - K_mean) + \
                   (T[I_i, J_j] + ex_val) * (K_i_prop[I_i] - K_mean) * (K_j_prop[J_j] - K_mean)
        pos_ex = (cor_plus > cor_minus)
        if r_diff > 0:
            T[I_i, I_j] += (pos_ex * 2 - 1) * ex_val
            T[J_i, J_j] += (pos_ex * 2 - 1) * ex_val
            T[J_i, I_j] -= (pos_ex * 2 - 1) * ex_val
            T[I_i, J_j] -= (pos_ex * 2 - 1) * ex_val
        else:
            T[I_i, I_j] -= (pos_ex * 2 - 1) * ex_val
            T[J_i, J_j] -= (pos_ex * 2 - 1) * ex_val
            T[J_i, I_j] += (pos_ex * 2 - 1) * ex_val
            T[I_i, J_j] += (pos_ex * 2 - 1) * ex_val
        r_diff = r - tn.generate.a_coef_from_matrix(T, i_prop, j_prop)
        print('r_diff', r_diff)

    return