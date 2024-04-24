from typing import List, Tuple
from scratch.statistics import correlation, standard_deviation, mean

def least_squares_fit(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x,y) * standard_deviation(y) /standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha
    when the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, x: List[float], y: List[float]) -> float:
    return sum((error(alpha, beta, x_i, y_i) ** 2) for x_i, y_i in zip(x,y))


from scratch.statistics import de_mean

def total_sum_of_squares(y: List[float]) -> float:
    """
    the total squared variation of y_i's from their mean
    """
    return sum(v ** 2 for v in de_mean(y))
    
def r_squared(alpha: float, beta: float, x: List[float], y: List[float]) -> float:
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))

