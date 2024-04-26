from typing import List
from scratch.statistics import daily_minutes_good
from scratch.gradient_descent import gradient_step


inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]


from scratch.linear_algebra import dot, Vector

def predict(x: Vector, beta: Vector) -> float:
    """ 
    Assumes that the first elements of x is 1
    """
    return dot(x, beta)

# 1. Error
from typing import List

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


# Loss function
def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


# Compute gradient of Loss function for each variable in x vector
def sqerror_gradient(x: Vector, y:  float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

# assert sqerror_gradient(x, y, beta) == [-12, -24, -36]

# 4. Gradient descent to find optimal beta

# Function will work with any dataset
import random
import tqdm
from scratch.linear_algebra import vector_mean
from scratch.gradient_descent import gradient_step

def least_squares_fit(xs: List[Vector],                  # rows = variables, column = for each y
                      ys: List[float],                   # y value vector of complete dataset
                      learning_rate: float = 0.001,      # step_size
                      num_steps: int = 1000,             # iterations
                      batch_size: int = 1                # stochastic
                     ) -> Vector:
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta).
    """
    
    # Start with a random guess of beta
    guess = [random.random() for _ in xs[0]]
    
    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]
            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess


from scratch.simple_linear_regression import total_sum_of_squares

def multiple_r_squared(xs: List[Vector], ys: List[float], beta: Vector) -> float:
    sum_of_squared_errors = sum([error(x, y, beta)**2 for x, y in zip(xs, ys)])
    return 1 - sum_of_squared_errors/ total_sum_of_squares(ys)

# assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68

from typing import TypeVar, Callable
X = TypeVar('X') # Generic type of data
Stat = TypeVar('Stat') # Generic type for "statistics"

# Generate one sample
def bootstrap_sample(data: List[X]) -> List[X]:
    """
    randomly samples len(data) elements with replacement
    """
    return [random.choice(data) for _ in data]


# Apply bootstraping on num_samples
# i.e. calculate stats_fn for all data_samples
def bootstrap_statistics(data: List[X], 
                         stats_fn: Callable[[List[X]], Stat], num_smaples: int) -> List[Stat]: 
    """
    evaluates stats_fn on num_samples bootstrap samples from data
    """
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_smaples)]

from scratch.probability import normal_CDF
def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
    # if the coefficient is positive, we need to compute twice the
    # probability of seeing an even *larger* value
        return 2 * (1 - normal_CDF(beta_hat_j / sigma_hat_j))
    else:
    # otherwise twice the probability of seeing a *smaller* value
        return 2 * normal_CDF(beta_hat_j / sigma_hat_j)

# assert p_value(30.58, 1.27) < 0.001 # constant term
# assert p_value(0.972, 0.103) < 0.001 # num_friends
# assert p_value(-1.865, 0.155) < 0.001 # work_hours
# assert p_value(0.923, 1.249) > 0.4 # phd

from typing import List
def ridge_penalty(beta: List[float], alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: List[float],
                        y: float,
                        beta: List[float],
                        alpha: float) -> float:
    """
    estimate error plus ridge penalty on beta
    """
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

from scratch.linear_algebra import add

def ridge_penalty_gradient(beta: List[float], alpha: float) -> List[float]:
    """gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: List[float],
                           y: float,
                           beta: List[float],
                           alpha: float) -> List[float]:
    """
    the gradient corresponding to the ith squared error term
    including the ridge penalty
    """
    return add(sqerror_gradient(x, y, beta),ridge_penalty_gradient(beta, alpha))

Vector = List[float]

def least_squares_fit_ridge(xs: List[Vector],                  # rows = variables, column = for each y
                      ys: List[float],  alpha,# y value vector of complete dataset
                      learning_rate: float = 0.001,      # step_size
                      num_steps: int = 1000,             # iterations
                      batch_size: int = 1                # stochastic
                     ) -> Vector:
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta).
    """
    
    # Start with a random guess of beta
    guess = [random.random() for _ in xs[0]]
    
    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]
            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess




