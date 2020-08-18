
### For long functions (>10 lines of code) not in a class (Semi-completed):

``` python
def ...(arg1, arg2=1):
    """
    Direct description of the function...
    
    Parameters
    ----------
    arg1 : string
      Description of arg1...
    arg2 : int, optional
      Description of arg2...
      Default=1.
    ...
    *args
      Variable length argument list.
    **kwargs
      Arbitrary keyword arguments.
    
    Returns
    -------
    int
      Description of return value...
    
    Raises
    ------
      ValueError: description of error types...
    
    Notes
    -----
     See https://...
     ...
    
    Examples
    --------
      Description of usage examples...
    """
    
    # Checks
    assert(...)
    
    # Initializations
    ...
        
    ... rest of the code ...
    
    
    return ...
```
    

### For long functions (>10 lines of code) not in a class (Empty):

``` python
def ...(...):
    """
    ...
    
    Parameters
    ----------
    ... : ...
      ...
    
    Returns
    -------
    ...
      ...
    
    Raises
    ------
      ...
    
    Notes
    -----
     ...
    
    Examples
    --------
      ...
    """
    
    # Checks
    assert(...)
    
    # Initializations
    ...
    
    return ...
```
    

### For shorts functions (<10 lines of code) not in a class (Empty):

``` python
def ...(...):
    """
    ...
    
    Parameters
    ----------
    ... : ...
      ...
    
    Returns
    -------
    ...
      ...
    """
    
    # Checks
    assert(...)
    
    # Initializations
    ...
    
    return ...
```
    

### For classes :

``` python
class ...(...):
    """
    ...
    
    Attributes
    ----------
    ... : ...
      ...
    
    
    """
    
    def __init__(self, arg1, arg2=1):
        """
        ...
        """
        self.a = ...
        self.b = ...
    
```


    
    