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


def matrix_from_sequence(K_in, K_out, c=0, i_prop='in', j_prop='out'):
    """ Chung Lu style adjacency matrix creation. Generating a target matrix T
    and compare it with a matrix filled with random variables. Assortativity
    may be induced by a non-zero c value.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
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

    delta_K_in = (K_in - K_in.mean()) / K_in.std()
    delta_K_out = (K_out - K_out.mean()) / K_out.std()
    delta_K_i_prop = eval('delta_K_'+i_prop)
    delta_K_j_prop = eval('delta_K_'+j_prop)

    T = (np.outer(K_in, K_out) / (N * K_in.mean())) * \
        (1 + c*np.outer(delta_K_i_prop, delta_K_j_prop))
    A = np.random.uniform(size=(N, N)) < np.clip(T, 0, 1)

    return A.astype(int)


def chung_lu_model(K_in, K_out, rho=0, c=0, i_prop='in', j_prop='out'):
    """ Compute an adjacency matrix from in- and out-degree sequences using the
    Chung Lu model.
    Create a

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    rho : float
        Correlation coefficient between in- and out-degree
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

    print('Adjacency matrix - Chug Lu model | N =', len(K_in), '| K_in = [',
          min(K_in), ',', max(K_in), '] | K_out = [', min(K_out), ',',
          max(K_out), '] | rho =', rho, '| c =', c)

    K_in.sort()

    if rho == 0:
        np.random.shuffle(K_out)
        A = matrix_from_sequence(K_in, K_out, c, i_prop=i_prop, j_prop=j_prop)

    elif rho > 0:
        K_out.sort()
        A = matrix_from_sequence(K_in, K_out, c, i_prop=i_prop, j_prop=j_prop)
        for i in range(1, len(K_in)):
            if correlation_from_matrix(A) < rho:
                break
            tn.generate.swap_pair(A)

    elif rho < 0:
        K_out[::-1].sort()
        A = matrix_from_sequence(K_in, K_out, c, i_prop=i_prop, j_prop=j_prop)
        for i in range(1, len(K_in)):
            if correlation_from_matrix(A) > rho:
                break
            tn.generate.swap_pair(A)

    print('    Successful.')

    return A


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