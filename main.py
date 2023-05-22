import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fun(x):
    return (x[0]-5) ** 2 + (x[1]-5) ** 2

def gradf(x):
    return  np.array([2*(x[0]-5),2*(x[1]-5)])

def constraint(x):
    return x[0] ** 2 + x[1] ** 2 - 2

def gradconstraint(x):
    return np.array([2*x[0],2*x[1]])

def directional_search(target_func, initial_point, direction, options=None):
    """
    Finds the intersection of a target function and a line in the specified direction starting from the initial point.

    :param target_func: Target function to intersect with line.
    :type target_func: function
    
    :param initial_point: Starting point for line intersection search.
    :type initial_point: numpy array
    
    :param direction: Direction of line for intersection search.
    :type direction: numpy array
    
    :param options: Optional dictionary of search options.
    :type options: dict
    
    :return: Dictionary containing search status, result, and number of iterations.
    :rtype: dict
    """

    # Set default search options
    if options is None:
        options = {}
    eps = options.get("eps", 1E-5)
    max_iter = options.get("max_iter", 1000)
    verbose = options.get("verbose", False)
    alpha = options.get("alpha", 1)
    iteration_data = options.get("iteration_data")

    # Initialize iteration data dictionary
    if iteration_data is None or not iteration_data or not verbose:
        logger.warning("Iteration data is None or empty. No data will be logged.")
    else:
        iteration_data_dict = {"alpha_list": [], "point_list": []}

    # Initialize flag for changing direction to negative
    changed_to_negative = False

    # Begin search iterations
    for ii in range(max_iter + 1):
        # Change direction to negative if target function is negative and direction has not been changed yet
        if target_func(initial_point) < 0 and not changed_to_negative:
            direction = -direction
            changed_to_negative = True
        # Change direction to positive if target function is positive and direction has been changed to negative
        elif target_func(initial_point) > 0 and changed_to_negative:
            direction = -direction
            changed_to_negative = False
        # Decrease alpha by 10% at each iteration
        alpha *= 0.9

        # Log current target value and alpha if verbose option is True
        if verbose:
            logger.info("Current target value g(x): %.4f at %s", target_func(initial_point), initial_point)

        # Return result if target value is within epsilon tolerance
        if np.abs(target_func(initial_point)) < eps:
            if verbose:
                logger.info("Intersection found at %s in %d iteration(s).", initial_point, ii)
            if iteration_data:
                return {"status": 1, "res": initial_point, "iter": ii, "iteration_data": iteration_data_dict}
            return {"status": 1, "res": initial_point, "iter": ii}

        # Update current point using alpha and direction
        initial_point = initial_point + alpha * direction

        # Log iteration data if iteration_data option is True
        if iteration_data:
            iteration_data_dict["point_list"].append(initial_point)
            iteration_data_dict["alpha_list"].append(alpha)

    # Return failure status if no intersection is found within max_iter iterations
    if verbose:
        logger.info("No solution was found.")
    return {"status": 0, "res": initial_point, "iter": ii}

def finite_difference(function, x: np.array([]), component: int = None, options: dict = {}):
    """
    Calculates the finite difference approximation of the gradient of a function at a given point.

    :param function: Function to calculate gradient of.
    :type function: function
    :param x: Point at which to calculate gradient.
    :type x: numpy array
    :param component: Component of gradient to return. Default is None, which returns entire gradient.
    :type component: int
    :param options: Additional options for finite difference calculation. Default is an empty dictionary.
    :type options: dict
    :raises ValueError: If component is not an integer or is not suitable for the underlying problem, or if h is zero.
    :return: Finite difference approximation of gradient at specified point.
    :rtype: numpy array
    """
    
    # Set default options
    h = options.get("h", 1E-6)
    schema = options.get("schema", "forward")
    verbose = options.get("verbose", 0)

    # Check dimension of gradient via x
    dim = x.shape[0]
    if component is not None:
        if not isinstance(component, int):
            logger.error(f"Value for component - {component} - is not an integer.")
            raise ValueError(f"Value for component - {component} - is not an integer.")
        if component < 0 or component > dim:
            logger.error(f"Value for component - {component} - is not suitable for underlying problem.")
            raise ValueError(f"Value for component - {component} - is not suitable for underlying problem.")
    if h == 0:
        logger.error(f"h can't be zero.")
        raise ValueError(f"h can't be zero.")
    if schema is None:
        logger.info("No schema for finite difference is set, using standard value, i.e, forward instead.")

    # Initialize finite difference approximation and h vector
    fd = np.zeros(dim)
    hvec = h * np.eye(dim)

    # Calculate finite difference approximation for each component of gradient
    for ii in range(dim):
        hcurr = np.atleast_1d(hvec[ii, :])
        if schema == "forward":
            fd[ii] = (function(x + hcurr) - function(x)) / h
        elif schema == "central":
            fd[ii] = (function(x + hcurr) - function(x - hcurr)) / (2 * h)
        if verbose > 0:
            logger.info(f"Calculate FDQ for component: {ii}")
            logger.info(f"Value:                       {fd[ii]}")
    
    # Return entire gradient or specified component
    if component is not None:
        return np.atleast_1d(fd[component])
    return fd



def minimize(x0,grad_f,constraint,constraint_grad,options=None):
    
    # Set default search options
    if options is None:
        options = {}
    iteration_data = options.get("iteration_data")

    # Initialize iteration data dictionary
    if iteration_data is None or not iteration_data:
        logger.warning("Iteration data is None or empty. No data will be logged.")
    else:
        iteration_data_list = []

    # Set default values for options
    TOL = options.get("tol", 1E-3)
    ITERMAX = options.get("itermax", 100)
    ALPHA = options.get("alpha", 0.1)
    verbose = options.get("verbose", False)
    
    # Set options for directional search
    options_directional_search = {"eps": 1E-4, "maxiter": 2000, "verbose": 0}
    
    # Set default values for gradient functions
    grad_f = grad_f or finite_difference
    grad_g = constraint_grad or finite_difference
    
    # Check if minimum has been found
    norm_of_constraint = np.linalg.norm(grad_g(x0), 2)
    norm_of_target = np.linalg.norm(grad_f(x0), 2)
    if np.linalg.norm(grad_f(x0)/norm_of_target + grad_g(x0)/norm_of_constraint, 2) <= TOL:
        return {"status": 1, "res": x0, "iter": ii}

    # Perform gradient descent
    for ii in range(ITERMAX+1):
        
        # Find intersection in direction of constraint gradient
        direction = -grad_g(x0) / norm_of_constraint
        x_intersection = directional_search(constraint, x0, direction, options_directional_search)

        # Update x0 based on intersection and gradient of objective function
        if x_intersection["status"]:
            x0 = x_intersection["res"]
            termination_value = np.linalg.norm(grad_f(x0)/norm_of_target- grad_g(x0)/norm_of_constraint, 2)
            if verbose:
                logger.info("Current termination value: %.4f", termination_value )            
            if termination_value <= TOL:
                return {"status": 1, "res": x0, "iter": ii, "iter_data":np.asarray(iteration_data_list)}
            x0 = x0 - ALPHA*grad_f(x0) /norm_of_target
            norm_of_constraint = np.linalg.norm(grad_g(x0), 2)
            norm_of_target = np.linalg.norm(grad_f(x0), 2)
            
            # Log iteration data if iteration_data option is True
            if iteration_data:
                iteration_data_list.append(x0)

        # Return failure if intersection not found
        else:
            return {"status": 0, "res": x0, "iter": ii}

def main():

    options = {"verbose" : 1}
    x = np.array([5.5,4.0])
    res = minimize(fun,x,gradf,constraint,gradconstraint,options)

if __name__ == "__main__":
    main()
