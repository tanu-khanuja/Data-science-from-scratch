import numpy as np
import matplotlib.pyplot as plt
import random
import math

SQRT_TWO_PI = math.sqrt(2 * math.pi)

#for all individual x values the probability is 1 between x=0 to 1
def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform RV is <=x"""
    if x < 0: return 0
    elif x < 1: return x
    else: return 1


def normal_PDF(x: float, mu: float=0, sigma: float=0):
    return (math.exp(-(x-mu)**2 / (2*(sigma**2)))/(sigma*SQRT_TWO_PI))


def normal_CDF( x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1+math.erf((x-mu)/(sigma*math.sqrt(2))))/2


#p: target probabilty for which x is to be determined
def inverse_normal_cdf(p: float, 
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 1e-5) -> float:
    """find approximate inverse of CDF using binary search"""
    #if PDF X is not standard distribution Z then we know [X = sigma*Z + mu]
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance = tolerance)

    #define range of distribution
    lo_z = -10
    hi_z = 10

    while hi_z - lo_z > tolerance:
        mid_z = (lo_z + hi_z) / 2  #find midpoint
        mid_p = normal_CDF(mid_z)
        if mid_p < p:
            lo_z = mid_z
        else:
            hi_z = mid_z

    return mid_z


def bernoulli_trial(p: float) -> int:   
    """to generate a random sample with specified p
    input: float
    output: int
    """
    return 1 if random.random() <p else 0  
# random.random() returns a random float number between 0 and 1 (excluding 1)


def binomial(p: float, n: int) -> int:
    """ returns the sum of n bernoulli trials"""
    return sum(bernoulli_trial(p) for _ in range(n)) 

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """pick binomials of n trials for every num_points and plot histogram"""
    data = [binomial(p,n) for _ in range(num_points)]
    # list of binomial sums for num_points number of group events
    data_counts = Counter(data)
    plt.bar([x for x in data_counts.keys()], [y/num_points for y in data_counts.values()], 0.8, color = '0.75')
    #lets plot the line chart with bar plot
    #we know the mean and SD for binomial distribution mu = n*p and SD = sqrt(n*p*(1-p))
    mu = n*p
    sigma = math.sqrt(n*p*(1-p))
    x_values = range(min(data), max(data)+1)
    y_values = [normal_PDF(i, mu, sigma) for i in x_values]
    plt.plot(x_values, y_values)
    plt.show()

