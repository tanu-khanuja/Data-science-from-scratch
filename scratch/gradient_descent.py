from scratch.linear_algebra import Vector, dot
def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v,v)


from typing import Callable
def difference_quotient(f: Callable[[float], float], 
                        x: float,
                        h: float) -> float:
    return (f(x+h)-f(x))/h


def square(x: float) -> float:
    return x*x

def derivative(x: float) -> float:
    return 2*x


from scratch.linear_algebra import Vector
from typing import List

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,                    #vector v consists variables (x,y..) of function f
                                                              #it is the point at which to find the difference qoutient
                                i: int,                       # i is index of variable in v wrt which we want to find PD 
                                h: float) -> float:           # step size
    """Returns the i-th partial difference quotient of f at v"""
    # for v_index, v_value in enumerate(v):
    #     if i == v_index:
    #         w = v_value + h
    #     else:
    #         w = v_value
    
    w = [v_value + (h if i == v_index else 0) for v_index, v_value in enumerate(v)]
    return (f(w) - f(v))/ h 
                             
#Estimate gradient

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      i: int,
                      h: float = 0.0001) -> List[float]:
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]  #returns a list of PD wrt individual variables



import random
from scratch.linear_algebra import scalar_multiply, add, distance

#lets define the gradient of equation : sum of squares e.g. f(x,y) = x^2 + y^2 + z^2
def sum_of_square_gradient(v: Vector) -> Vector:
    return [2*i for i in v]

#lets define new gradient step towards optimal value
def gradient_step(v: Vector, gradient: Vector, step_size: float) ->  Vector:  #returns new v-value reaching towards optimal value
                                                                              #we input gradient of our function
                                                                              #position vector v which is to be minimized
                                                                              #step_size is different from 'h' used above
                                                                              #step_size is the increment parameter of gradient descent algorithm to move towards optimal solution
    assert len(v) == len(gradient)  #check if gradient for each v element is present
    increment = scalar_multiply(gradient, step_size)
    return add(v, increment)

#def gradient of loss function
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta   #update slope and intercept for updated theta
    predicted = slope * x + intercept
    error = predicted - y
    return [2 * error * x, 2 * error]


def minibatches(dataset: List[T],  
                batch_size: int,                             # to 
                shuffle: bool = True) -> Iterator[List[T]]:  #iterator type can be iterated using loops 
    """Generates "batch_size" sized minibatches from the dataset"""
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    
    if shuffle: random.shuffle(batch_starts)

    for start in batch_starts:
        end = start+batch_size
        yield dataset[start:end]
