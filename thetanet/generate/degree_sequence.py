import numpy as np
import thetanet as tn


def degree_sequence_single(k, P_k, N):
    """ Create a degree sequence. I.e. sample N values from k with
    probability P_k.

    Parameters
    ----------
    k : ndarray, 1D int
        Degree space.
    P_k : ndarray, 1D float
        Degree probability.
    N : int
        Number of neurons.

    Returns
    -------
    K : ndarray, 1D int
        Degree sequence.
    """

    x = np.cumsum(P_k)  # cumulative sum of the distribution
    x = np.append(np.zeros(1), x)  # add a zero at the beginning
    X = np.random.uniform(x[0], x[-1], size=N)  # N random values from cumsum
    k = np.append(k, k[-1] + 1)
    K = np.floor(np.interp(X, x, k))  # get N degrees, according to X

    return K.astype(int)


def degree_sequences_match(k_in, P_k_in, k_out, P_k_out):
    """ Check whether the number of in- and out-connections (stubs) can lead
    to a network by comparing mean degree values.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    P_k_in : ndarray, 1D float
        In-degree probability.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k_out : ndarray, 1D float
        Out-degree probability.

    Returns
    -------
    Boolean value.
    """

    k_in_mean = np.dot(k_in, P_k_in)  # average value of node degrees
    k_out_mean = np.dot(k_out, P_k_out)
    if abs(k_in_mean - k_out_mean) < 1:
        return True
    else:
        return False


def degree_sequence_double(k_in, P_k_in, k_out, P_k_out, N, console_output=True):
    """ Generate two degree sequences that match and can be wired up to form
    a network.
    1) Check if mean values of degrees do match.
    2) Randomly generate in-degree sequence.
    3) Randomly generate out-degree sequence and check if the difference in
    degrees is not too large (<5% of N).
    4) Deterministically change the out-degree sequence.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    P_k_in : ndarray, 1D float
        In-degree probability.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k_out : ndarray, 1D float
        Out-degree probability.
    N : int
        Number of neurons.

    Returns
    -------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    """

    if not degree_sequences_match(k_in, P_k_in, k_out, P_k_out):
        print('| ERROR: Change degree distributions to fulfill: '
              'k_in_mean = k_out_mean')
        exit(1)

    d = None  # Difference between the sums of sequences
    i = 0  # loop count
    K_in = degree_sequence_single(k_in, P_k_in, N)
    while d != 0:
        i += 1
        K_out = degree_sequence_single(k_out, P_k_out, N)
        d = abs(K_in.sum() - K_out.sum())
        if d < 0.01 * N:  # as soon as d is small
            match_degree_sequences(k_in, K_in, k_out, K_out)
            break

    if console_output:
        print('| Found sequences after', i,
              'attempts.')
        print('| Difference in degrees: ', d)

    return K_in, K_out


def match_degree_sequences(k_in, K_in, k_out, K_out):
    """ Adjust both degree-sequences to make them match.
    Let the difference in connections be d. We raise one sequence by d/2
    and lower the other by d/2. The respective neurons are picked randomly
    which preserves the degree probabilities.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    K_in : ndarray, 1D int
        In-degree sequence.
    k_out : ndarray, 1D int
        Out-degree space.
    K_out : ndarray, 1D int
        Out-degree sequence.

    Returns
    -------
    K_in, K_out will be modified.
    """

    def adjust_sequence(K, k, d):
        """ Adjust a given degree sequence by the number d. Those neurons with
        already extreme degree values according to the degree space will not
        be used to do that.
        NOTE: Distribution get skewed towards the higher/lower end! Let d be
        quite small.

        Parameters
        ----------
        K : ndarray, 1D int
            Degree sequence.
        k : ndarray, 1D int
            Degree space.
        d : int
            Number of connection to be adjusted. The sign of d indicates in
            which direction.

        Returns
        -------
        K will be modified.
        """

        if d > 0:
            K_notMin = K[K != min(k)]
            tune_down = np.random.choice(len(K_notMin), abs(d), replace=False)
            K_notMin[tune_down] -= 1
            K[K != min(k)] = K_notMin
        else:
            K_notMax = K[K != max(k)]
            tune_up = np.random.choice(len(K_notMax), abs(d), replace=False)
            K_notMax[tune_up] += 1
            K[K != max(k)] = K_notMax

            return

    num_edges = int((sum(K_in) + sum(K_out)) / 2)  # round to smaller number
    d_in = sum(K_in) - num_edges
    d_out = sum(K_out) - num_edges
    adjust_sequence(K_in, k_in, d_in)
    adjust_sequence(K_out, k_out, d_out)

    return


def correlate_sequences(K_in, K_out, rho):
    """ Correlate in- and out-degrees with correlation rho.
    We can always sort one sequence, e.g. the in-degree sequence. Since it is
    faster to destroy correlation than to build it up we start with extreme
    correlation and swap out-degrees of randomly selected neuron pairs until
    the desired correlation is reached.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    rho : float
        In-/out-degree correlation.

    Returns
    -------
    K_in, K_out will be modified.
    """

    K_in.sort()
    N = len(K_in)

    d_rho = rho_from_sequences(K_in, K_out) - rho
    while abs(d_rho) > 0.001:
        n = np.random.randint(0, N-1, 2)
        K_out[n] = np.flipud(K_out[n])
        d_trail = rho_from_sequences(K_in, K_out) - rho
        if abs(d_trail) < abs(d_rho):
            d_rho = d_trail
        else:
            K_out[n] = np.flipud(K_out[n])

    return


def rho_from_sequences(K_in, K_out):
    """ Compute Pearson correlation coefficient for two given sequences.

    Parameters
    ----------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.

    Returns
    -------
    rho : float
        Pearson correlation coefficient.
    """

    rho = np.mean((K_in - K_in.mean()) * (K_out - K_out.mean())) \
          / np.std(K_out) / np.std(K_in)

    return rho


def swap_pair(x):
    """ Swap two entries in first dimension of an array.
    Application in degree sequence (scalars) or in adjacency matrix (rows)

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    x will be modified.
    """

    random_pair = np.random.choice(x.shape[0], 2)
    x[random_pair] = np.flipud(x[random_pair])

    return


def degree_sequence(k_in, P_k_in, k_out, P_k_out, N, rho=0,
                    console_output=True):
    """ Generate a degree sequence for a directed network.
    In- and out-degrees are drawn with probability P_k from the space k and
    subsequently correlated so they have a Pearson correlation coefficient
    of rho.

    Parameters
    ----------
    k_in : ndarray, 1D int
        In-degree space.
    P_k_in : ndarray, 1D float
        In-degree probability.
    k_out : ndarray, 1D int
        Out-degree space.
    P_k_out : ndarray, 1D float
        Out-degree probability.
    N : int
        Number of neurons.
    rho : float
        In-/out-degree correlation.
    console_output: bool
        Whether or not to print details to the console.

    Returns
    -------
    K_in : ndarray, 1D int
        In-degree sequence.
    K_out : ndarray, 1D int
        Out-degree sequence.
    """

    if console_output:
        print('\nCreating degree sequence (swap)')
        print('|......................................')
        print('| N =', N, 'and N_edges =', np.round(N*k_in.dot(P_k_in), 2))
        print('|......................................')

    K_in, K_out = degree_sequence_double(k_in, P_k_in, k_out, P_k_out, N,
                                         console_output=console_output)
    correlate_sequences(K_in, K_out, rho)

    return K_in, K_out


def degree_sequence_copula(P_k, N, k_in, k_out, console_output=True):
    """ Generate two degree sequences that match and can be wired up to form
        a network.
        1) Check if mean values of degrees do match.
        2) Randomly generate in- and out-degree sequence and check if the
        difference in degrees is not too large (<1% of N).
        3) Deterministically change the out-degree sequence.

        Parameters
        ----------
        P_k : ndarray, 2D float
            Joint In-Out-degree probability.
        N : int
            Number of neurons.
        k_in : ndarray, 1D int
            In-degree space.
        k_out : ndarray, 1D int
            Out-degree space.
        console_output: bool
            Whether or not to print details to the console.

        Returns
        -------
        K_in : ndarray, 1D int
            In-degree sequence.
        K_out : ndarray, 1D int
            Out-degree sequence.
    """

    if console_output:
        print('\nCreating degree sequence (copula)')
        print('|......................................')
        print('| N =', N, 'and N_edges =', np.round(N*k_in.dot(P_k.sum(1)),2))
        print('|......................................')

    if not degree_sequences_match(k_in, P_k.sum(1), k_out, P_k.sum(0)):
        print('| ERROR: Change degree distributions to fulfill: '
              'k_in_mean = k_out_mean')
        exit(1)

    d = None  # Difference between the sums of sequences
    i = 0  # loop count
    while d != 0:
        i += 1
        K_in, K_out = tn.utils.sample_from_bivar_pdf(P_k, N, k_in, k_out)
        d = abs(K_in.sum() - K_out.sum())
        if d < 0.01 * N:  # as soon as d is small
            match_degree_sequences(k_in, K_in, k_out, K_out)
            break

    if console_output:
        print('| Found sequences after', i,
              'attempts.')
        print('| Difference in degrees: ', d)

    return K_in, K_out