from scratch.linear_algebra import vector_mean
from scratch.statistics import standard_deviation
import math
from typing import Tuple, List
Vector = List[float]

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """
    returns the mean and standard deviation for each position
    """
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    return means, stdevs

# We can then use them to create a new dataset:

def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position
    as is if its standard deviation is 0.)
    """
    dim = len(data[0])
    means, stdevs = scale(data)
    
    # Make a copy of each vector
    rescaled = [v[:] for v in data] 
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled