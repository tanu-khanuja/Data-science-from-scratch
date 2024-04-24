from typing import List, Tuple, Callable
import math
Vector = List[float]  
Matrix = List[Vector]

import math

def add(a: Vector, b: Vector) -> Vector:
    """Adds corresponding elements"""
    
    assert len(a)==len(b), "Input vectors must be of same length"
    
    return [(a_i + b_i) for a_i, b_i in zip(a, b)]

def subtract(a: Vector, b: Vector) -> Vector:
    """Subtracts corresponding elements"""

    assert len(a) == len(b)

    return [(a_i - b_i) for a_i, b_i in zip(a,b)]

def vector_sum(list_of_vectors: List[Vector]) -> Vector:
    """sum of all corresponding elements"""
   
    #check if list_of_vectors is empty
    #assert list_of_vectors, "no vectors provided"

    #check if vectors are of same size
    l = len(list_of_vectors[0])  #length of first vector
    #all() returns True if all elements in given iterable are true
    assert all(len(v)==l for v in list_of_vectors), "vectors are of different sizes!" 

    return [sum(v[i] for v in list_of_vectors) for i in range(l)]

def scalar_multiply(v:Vector, c:float) -> Vector:
    """multiplies every element by c"""
    l = len(v)
    return([c*v[i] for i in range(l)])

def vector_mean(v: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    a = vector_sum(v)
    return scalar_multiply(a, 1/len(v))

def dot(a: Vector, b: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(a)==len(b), "different sizes"
    l = len(a)
    return(sum(a[i]*b[i] for i in range(l)))


def sum_of_squares(a: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(a, a)

def magnitude(a: Vector):
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(a))

def squared_distance(a: Vector, b: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(a,b))


def distance(a: Vector, b: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(a,b))


def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    n_rows = len(A)
    n_col = len(A[0]) if A else 0  
    return (n_rows, n_col)



def get_row(A: Matrix, i: int) -> List:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]


def get_column(A: Matrix, j: int) -> List:
    """Returns the j-th column of A (as a Vector)"""
    return [r[j] for  r in A]

def make_matrix(num_rows: int,
                num_cols: int, 
                entry_fn: Callable[[int, int],float]) -> Matrix:
    """
Returns a num_rows x num_cols matrix
whose (i,j)-th entry is entry_fn(i, j)
"""
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(size: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(size, size, lambda i, j: 1 if i==j else 0)
