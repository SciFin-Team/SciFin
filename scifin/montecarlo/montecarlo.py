# Created on 2020/8/11

# This module is for generating Monte Carlo simulations.

# Standard library imports
from typing import Any, Callable

# Third party imports
from typeguard import typechecked

# Local application imports
from .. import timeseries as ts


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

@typechecked
def generate_series(n: int=1,
                    generate_names: bool=True,
                    names_base: str="rs",
                    series_model: Callable[[], ts.TimeSeries]=None,
                    **kwargs: Any
                    ) -> list:
    """
    Generate a list of `n` series of the type `series_model`.
    Here all series have the same building parameters.
    
    Parameters
    ----------
    n : int
      Number of time series to be generated.
    generate_names : bool
      Option to generate names.
    names_base : str
      Base for the names.
    series_model : function
      TimeSeries generating function.
    **kwargs
        Arbitrary keyword arguments.
      
    Returns
    -------
    List of TimeSeries
      The n time series that were generated.
    """
    
    # Checks
    assert(isinstance(n,int))

    # Create list
    L = []
    for i in range(n):
        if generate_names is True:
            L.append(series_model(name=names_base+str(i), **kwargs))
        else:
            L.append(series_model(**kwargs))
    
    return L



#---------#---------#---------#---------#---------#---------#---------#---------#---------#