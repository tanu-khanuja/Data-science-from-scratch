from typing import Tuple  #to get mean and SD in a tuple
import math               #to find sqrt in eqn
import random
from typing import List

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """returns mu and sigma corresponding to binomial(n,p)"""
    mu = n*p
    sigma = math.sqrt(n*p*(1-p))
    return (mu, sigma)


from scratch.probability import normal_CDF
#normal CDF is probability for an event less than a threshold x
normal_probability_below = normal_CDF

#probability above threshold x
def normal_probability_above(lo: float,
                             mu: float=0,
                             sigma: float=1) -> float:
    "probability that an N(mu, sigma) > lo(threshold)"""
    return 1-normal_CDF(lo, mu, sigma)


#probability between lo and hi
def normal_probability_between(lo:float, hi:float, mu:float=0, sigma:float=1) -> float:
    """probability than an N(mu, sigma) is between lo and hi"""
    return normal_CDF(hi,mu,sigma) - normal_CDF(lo,mu,sigma)

#probability outside lo and hi
def normal_probability_outside(lo:float, hi:float, mu:float=0, sigma:float=1) -> float:
    """probability that an N(mu,sigma) is not between lo and hi"""
    return 1-normal_probability_between(lo,hi,mu,sigma)



from scratch import probability as pb

def normal_upper_bound(probability:float, 
                       mu:float=0,
                       sigma:float=1) -> float:
    """Returns z for which P(X<=z)=probability"""
    return pb.inverse_normal_cdf(probability,mu,sigma)


#calculate p-value using 'two-sided test' methods

def two_sided_p_value(x: float, mu: float=0, sigma: float=1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu, sigma)?
    """
    if x>=mu:
        # tail is all probability above x
        #doubled to take both sides of mean
        return 2*normal_probability_above(x, mu, sigma)
        # Why did we use a value of 529.5 rather than using 530? 
        # This is called a **continuity correction.  
        # It reflects the fact that normal_probability_between(529.5,530.5, mu_0, sigma_0) 
        # is a better estimate of the probability of seeing 530 heads than normal_probability_between(530, 531, mu_0, sigma_0) is.
    else:
        #if x<mean, tail is all below x
        #doubled to tak both sides of mean
        return 2*normal_probability_below(x, mu, sigma)


def normal_lower_bound(probability:float, 
                       mu:float=0,
                       sigma:float=1) -> float:
    """Returns z for which P(X>=z) = probability"""
    return pb.inverse_normal_cdf(1-probability,mu,sigma)

def normal_two_sided_bounds(probability:float, 
                           mu:float=0,
                           sigma:float=1) -> Tuple[float,float]:
    """returns symmetric bounds(around mean) that contains specified probability"""
    tail_probability = (1-probability)/2
    lower_bound = normal_upper_bound(tail_probability,mu,sigma)
    upper_bound = normal_lower_bound(tail_probability,mu,sigma)
    return (lower_bound,upper_bound)


#simulate n flip of a coin
def coin_flip (n: int) -> List[int]:
    return[random.randint(0,1) for _ in range(n)]


def run_experiment() -> List[bool]:
    """flips a fair coin 1000 times. True = heads, False = tails"""
    return (random.random()< 0.5 for _ in range(1000))   #generate list of true/false for 1000 coin flips

random.seed(0)


def reject_fairness(exp: List[bool]) -> bool:
    """using the 5% significance levels """
    num_heads = len([flip for flip in exp if flip])
    return num_heads<469 or num_heads>531   #returns true/false


def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n/N
    return N*p, math.sqrt(p*(1-p)*N)

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

def B(alpha: float, beta: float) -> float:
    """A normalizing constant so that the total probability is 1"""
    return math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha+beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x<=0 or x>=1:
        return 0
    else:
        return x**(alpha-1) * (1-x)**(beta-1) / B(alpha, beta)
