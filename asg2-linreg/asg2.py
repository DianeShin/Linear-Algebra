"""asg2.py

This assignment is based on the Linear Regression - Least Squares
theory.  The present script is associated with the text document
available from the course web page.

Do not touch the imports, and specifically, do not import matplotlib
in this file! Use the provided file draw.py for visualization. You can
run it by executing the script from command line:

python3 draw.py


The imports listed below should be enough to accomplish all tasks.

The functions /docstring/s contain some real examples and usage of the
functions. You can run these examples by executing the script from
command line:

python3 asg2.py

Note that the unit tests for the final grading may contain different
tests, and that certain requirements given below are not tested in the
testing before the final testing.
"""

import numpy as np
from scipy import linalg as la

np.set_printoptions(precision=3)

def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b.
    You should do this by using the QR decomposition.

    YOUR ARE NOT ALLOWED TO USE THE FUNCTION lstsq() FROM NUMPY or SCIPY 

    Parameters:
        A ((m,n) ndarray): A matrix 
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.

    Examples
    --------
    >>> A = np.array([[4, 1],[1, 1],[8, 9],[6, 9],[5, 2],[7, 7],[7, 1],[5, 1]])
    >>> b = np.array([[8],[4],[1],[8],[4],[6],[7],[8]])
    >>> np.array(least_squares(A,b))
    array([[ 1.236],
           [-0.419]])
    """
    Q, R = la.qr(A, mode = 'economic')
    return la.solve_triangular(R, np.dot(np.transpose(Q), b))

def linear_model(x,y):
    """Find the a and b coefficients of the least squares line y = ax + b.

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the line y = ax + b.

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])
    >>> np.array(linear_model(x,y))
    array([0.095, 1.621])
    """
    return least_squares(np.vstack([x, np.ones(x.size)]).T, y)

def exponential_model(x,y):
    """Find the a and b coefficients of the best fitting curve y = ae^(bx).

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the model  y = ae^(bx).

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])        
    >>> np.array(exponential_model(x,y))
    array([1.662, 0.045])
    """
    arr = linear_model(x, np.log(y))
    return np.array([np.exp(arr[1]), arr[0]])



def power_model(x,y):
    """Find the a and b coefficients of the best fitting curve y = a x^b.

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the model  y = a x^b.

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])     
    >>> np.array(power_model(x,y))
    array([1.501, 0.219])
    """
    
    arr = linear_model(np.log(x), np.log(y))
    return np.array([np.exp(arr[1]), arr[0]])



def logarithmic_model(x,y):
    """Find the a and b coefficients of the best fitting curve y = a + b ln x.

    Parameters
    ----------
    x       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    y       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    (a,b)   : a tuple containing the coefficients of the model y = a + b ln x.

    Examples
    --------
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])     
    >>> np.array(logarithmic_model(x,y))
    array([1.415, 0.455])
    """
    
    return linear_model(np.log(x), y)[::-1]



def training_error(f,xx,yy):
    """Find the sum of squared errors of the model f on the data xx and yy used to
    determine the parameters of f.

    Parameters
    ----------
    f        : a lambda function containing the fitted parameters 
               implementing one models under study
    xx       : np.ndarray : a numpy array of floats for the input (predictor variables) 
    yy       : np.ndarray : a numpy array of floats for the output (response variable)
    
    Returns
    -------
    err      : a float representing the training error.
    >>> x = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([1.75, 1.91, 2.03, 2.13, 2.22, 2.30, 2.37, 2.43])     
    >>> a,b = power_model(x,y)
    >>> f_pow = lambda xx: a*(xx**b)
    >>> np.array(training_error(f_pow, x, y))
    array(0.008)
    """
    return np.sqrt(np.sum(np.square(f(xx) - yy)))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
