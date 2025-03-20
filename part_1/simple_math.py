"""
A collection of simple math operations
"""

def simple_add(a,b):
    """
    Return the result of adding two numbers.

    Parameters
    ----------
    a : float
        The first number to add.
    b : float
        The second number to add.
    
    Returns
    -------
    float
        The result of adding a and b.
    """
    return a+b

def simple_sub(a,b):
    """
    Return the result of subtracting one number from another.

    Parameters
    ----------
    a : float
        The number to subtract from.
    b : float
        The number to subtract.
    
    Returns
    -------
    float
        The result of subtracting b from a.
    """
    return a-b

def simple_mult(a,b):
    """
    Return the result of multiplying two numbers.

    Parameters
    ----------
    a : float
        The first number to multiply.
    b : float
        The second number to multiply.
    
    Returns
    -------
    float
        The result of multiplying a and b.
    """
    return a*b

def simple_div(a,b):
    """
    Return the result of dividing one number by another.

    Parameters
    ----------
    a : float
        The number to be divided.
    b : float
        The number to divide by.
    
    Returns
    -------
    float
        The result of dividing a by b.
    """
    return a/b

def poly_first(x, a0, a1):
    """
    Return the evaluated value of a first degree polynomial: p(x) = a0 + a1 * x.

    Parameters
    ----------
    x : float
        The value at which to evaluate the polynomial.
    a0 : float
        The coefficient of x^0.
    a1 : float
        The coefficient of x^1.
    
    Returns
    -------
    float
        The result of evaluating p(x) = a0 + a1 * x.
    """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """
    Return the evaluated value of a second degree polynomial: p(x) = a0 + a1 * x + a2 * x^2.

    Parameters
    ----------
    x : float
        The value at which to evaluate the polynomial.
    a0 : float
        The coefficient of x^0.
    a1 : float
        The coefficient of x^1.
    a2 : float
        The coefficient of x^2.
    
    Returns
    -------
    float
        The result of evaluating p(x) = a0 + a1 * x + a2 * x^2.
    """
    return poly_first(x, a0, a1) + a2*(x**2)