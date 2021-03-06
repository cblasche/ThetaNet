from scipy.stats import multivariate_normal
from scipy import interpolate
import numpy as np


def bivar_pdf(pdf_y, pdf_z, rho_gauss):
    """ Create correlated bi-variate probability density function (pdf) of
    two pdf's with arbitrary marginal.
    This approach is based on mapping each variable y and z to a normal
    distributed variable x. These mappings will be applied to a correlated
    bi-variate Gaussian cdf to gain a joint cdf in y-z space. The desired
    pdf will be constructed from this cdf.

    Parameters
    ----------
    pdf_y : ndarray, 1D float
        Probability density function of first variable.
    pdf_z : ndarray, 1D float
        Probability density function of second variable.
    rho_gauss : float, ]-1 ... 1[
        Covariance of bi-variate normal probability density function.
        Note: The correlation of samples from the output pdf is different.

    Returns
    -------
    pdf_yz : ndarray, 2D float
        Joint correlated pdf.
    """

    def map_indices_to_x(pdf_y, x, pdf_x):
        """ Compute variables x which refer to the indices of y.
        This can be done based on their cumulative density function (cdf) and
        their inverse.

        Parameters
        ----------
        pdf_y : ndarray, 1D float
            Probability density function of random variable y.
        x : ndarray, 1D float
            Space of random variable x.
        pdf_x :
            ndarray, 1D float
            Probability density function of random variable x.
        Returns
        -------
        x(y_ind) : ndarray, 1D float
            Respective points in x space, which have same percentile as
            y(y_ind).
        """

        def stretch_cdf(cdf):
            """ Stretch the function to the interval [0, 1].
            This ensures a complete mapping.
            Note: No return value, the cdf gets modified in place.
            """
            cdf -= cdf.min()
            cdf /= cdf.max()

        # Adding a zero to the beginning:
        # Before integrating over the first interval the cumulative
        # probability is 0.
        # Also kind of needed for the np.diff operation at the end, which
        # shrinks the array by 1.
        cdf_y = np.append(0, pdf_y.cumsum(0))
        cdf_x = pdf_x.cumsum(0)
        stretch_cdf(cdf_y)
        stretch_cdf(cdf_x)

        # Indices of y; y is not necessarily needed.
        y_ind = np.arange(len(cdf_y))

        # Inverse cumulative density of y: y as a function of percentile
        y_ind_percentile = interpolate.interp1d(cdf_y, y_ind, kind='cubic')

        # Mapping from x to y_ind:
        # I.e.: x to percentile (This is the cdf_x) and from percentile space
        # to y_ind via the function y_ind_percentile. This would be:
        #
        #   y_ind_of_x = interpolate.interp1d(x, y_ind_percentile(cdf_x))
        #
        # But we need the inverse of it to map from y_ind to x.
        x_of_y_ind = interpolate.interp1d(y_ind_percentile(cdf_x), x,
                                          kind='cubic')

        return x_of_y_ind(y_ind)

    # Parameters and domain of normal distribution function
    mean = 0
    var = 1
    covar = rho_gauss
    b = 8*np.sqrt(var)  # Domain boundary till where normal pdf will be computed
    x = np.linspace(-b+mean, b+mean, 500)
    x0, x1 = np.meshgrid(x, x)
    xx = np.stack([x0, x1], axis=2)

    # Pdf of bi-variate normal distribution
    pdf_x_2D = multivariate_normal([mean, mean],
                                   [[var, covar], [covar, var]]).pdf(xx)
    pdf_x_2D /= pdf_x_2D.sum()
    pdf_x = pdf_x_2D.sum(1)

    # Cdf of bi-variate normal distribution
    cdf_x_2D = pdf_x_2D.cumsum(0).cumsum(1)
    cdf_x_2D_func = interpolate.RectBivariateSpline(x, x, cdf_x_2D)

    # Target cdf through remapping
    y_mapped = map_indices_to_x(pdf_y, x, pdf_x)
    z_mapped = map_indices_to_x(pdf_z, x, pdf_x)
    cdf_yz = cdf_x_2D_func(y_mapped, z_mapped)

    # Target pdf
    pdf_yz = np.diff(np.diff(cdf_yz, axis=0), axis=1)

    # Ensure probability properties
    pdf_yz = np.clip(pdf_yz, 0, 1)
    pdf_yz /= pdf_yz.sum()

    return pdf_yz


def sample_from_bivar_pdf(pdf, N, x0=None, x1=None):
    """ Create 2 lists of N indices with probability according to the target
    probability density function (pdf)

    Parameters
    ----------
    pdf : ndarray, 2D float
        Probability density function of 2D variable (x0, x1).
    N : int
        Sample size.
    x0 : ndarray, 1D float
        Space of random variable x0.
    x1 : ndarray, 1D float
        Space of random variable x1.

    Returns
    -------
    ind0 / x0[ind0]: ndarray, 1D int
        List of indices for first variable or variables themselves.
    ind1 / x1[ind1]: ndarray, 1D int
        List of indices for second variable or variables themselves.
    """

    n0 = pdf.shape[0]
    n1 = pdf.shape[1]
    index = np.random.choice(np.arange(n0*n1), N, p=pdf.ravel())
    ind0 = index % n0
    ind1 = np.floor_divide(index, n0)

    if x0 is None or x1 is None:
        return ind0, ind1
    else:
        return x0[ind0], x1[ind1]


def correlation_from_pdf(pdf, x0, x1):
    """ Compute Pearson correlation coefficient for given joint pdf of random
    variables x0 and x1.

    Parameters
    ----------
    pdf : ndarray, 2D float
        Probability density function of 2D variable (x0, x1).
    x0 : ndarray, 1D float
        Space of random variable x0.
    x1 : ndarray, 1D float
        Space of random variable x1.

    Returns
    -------
    rho : float
        Pearson correlation coefficient.
    """

    pdf_x0 = pdf.sum(1)
    pdf_x1 = pdf.sum(0)
    x0_mean = x0.dot(pdf_x0)
    x1_mean = x1.dot(pdf_x1)

    cov = (np.outer(x0-x0_mean, x1-x1_mean) * pdf).sum()
    std_x0 = np.sqrt((pdf_x0 * (x0 - x0_mean) ** 2).sum())
    std_x1 = np.sqrt((pdf_x1 * (x1 - x1_mean) ** 2).sum())
    rho = cov / std_x0 / std_x1

    return rho


def bivar_pdf_rho(y, pdf_y, z, pdf_z):
    """ Create a function which computes the joint pdf for y and z given the
    the correlation rho.
    Can be called like that:
        pdf_yz = bivar_pdf_rho(y, pdf_y, z, pdf_z)(rho)
    In contrast to bivar_pdf(pdf_y, pdf_z, rho_gauss) where the correlation
    of the gauss function is set!

    Parameters
    ----------
    y : ndarray, 1D int
        In-degree space.
    pdf_y : ndarray, 1D float
        In-degree probability.
    z : ndarray, 1D int
        Out-degree space.
    pdf_z : ndarray, 1D float
        Out-degree probability.

    Returns
    -------
    pdf_yz_rho : func
        Takes one argument (rho) and returns pdf_yz.
    """

    def rho_from_rho_gauss(rho_gauss):
        pdf_yz = bivar_pdf(pdf_y, pdf_z, rho_gauss=rho_gauss)
        return correlation_from_pdf(pdf_yz, y, z)

    rho_gauss_arr = np.linspace(-0.99, 0.99, 10)
    rho_arr = [rho_from_rho_gauss(rho_gauss) for rho_gauss in rho_gauss_arr]

    def pdf_yz_rho(rho):
        try:
            rho_gauss = interpolate.interp1d(rho_arr, rho_gauss_arr, kind='cubic')(rho)
        except ValueError:
            print('\nCorrelation rho =', rho, 'is out of the interpolation'
                  ' range. Try smaller absolute values.')
            quit(1)
        pdf_yz = bivar_pdf(pdf_y, pdf_z, rho_gauss=rho_gauss)
        return pdf_yz

    return pdf_yz_rho
