# Created on 2020/8/14

# This module is for storing built-in classes for exceptions.


class AccessError(Exception):
    """Raised when a file or url cannot be accessed."""
    pass
    

class SamplingError(Exception):
    """Raised when the sampling of a time series is not uniform."""
    pass


class ArgumentsError(Exception):
    """Raised when provided arguments of a function are not satisfactory."""
    pass


