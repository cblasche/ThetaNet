class ConvergenceError(Exception):
    """ Raised when scheme does not converge.
    """
    pass


class NoPeriodicOrbitException(Exception):
    """ Raised when integration does not lead to an orbit.
    """
    pass
