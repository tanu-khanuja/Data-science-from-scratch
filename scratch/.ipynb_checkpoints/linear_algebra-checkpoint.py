from typing import List, Tuple, Callable #for lists' type annotation
import math

Vector= List[float]     #define 'Vector' annotation
Matrix = List[List[float]]


def add(a: Vector, b: Vector) -> Vector:
    
    assert len(a)==len(b), "Input vectors must be of same length"
    
    return [(a_i + b_i) for a_i, b_i in zip(a, b)]

assert add([3,4],[2,1]) == [5,5]
# assert add([2,3],[3,5,3]) == [5,8] #Will generate assertion error, not same length



def subtract(a: Vector, b: Vector) -> Vector:

    assert len(a) == len(b)

    return [(a_i - b_i) for a_i, b_i in zip(a,b)]

assert subtract([1,2],[3,4]) == [-2,-2]



def component_add(list_of_vectors: List[Vector]) -> Vector:
   
    #check if list_of_vectors is empty
    assert list_of_vectors, "no vectors provided"

    #check if vectors are of same size
    l = len(list_of_vectors[0])  #length of first vector
    #all() returns True if all elements in given iterable are true
    assert all(len(v)==l for v in list_of_vectors), "vectors are of different sizes!" 

    return [sum(v[i] for v in list_of_vectors) for i in range(l)]


def scalar_multiply(v:Vector, c:float) -> Vector:
    l = len(v)
    return([c*v[i] for i in range(l)])


def component_mean(v: List[Vector]) -> Vector:
    num_of_vectors = len(x)
    l = len(x[0])
    return [(sum(v[i] for v in x)*(1/number_of_vectors)) for i in range(l)]

def component_mean(v: List[Vector]) -> Vector:
    a = component_add(v)
    return scalar_multiply(a, 1/len(v))


def dot_product(a: Vector, b: Vector) -> float:
    assert len(a)==len(b), "different sizes"
    l = len(a)
    return(sum(a[i]*b[i] for i in range(l)))


def sum_of_squares(a: Vector) -> float:
    l = len(a)
    sum_a = sum(math.pow(a[i],2) for i in range(l))
    return sum_a


def magnitude(a: Vector):
    return math.sqrt(sum_of_squares(a))
    
def squared_distance(a: Vector, b: Vector) -> float:
    return sum_of_squares(subtract(a,b))

def distance(a: Vector, b: Vector) -> float:
    assert len(a) == len(b), "Different sizes"
    l = len(a)
    difference = subtract(a,b)   #call subtract function 
    sq_vector = [math.pow(difference[i],2) for i in range(l)]
    sqrt_of_sum = math.sqrt(sum(sq[i] for i in range(l)))
    return (sqrt_of_sum)


def shape(A: Matrix) -> Tuple[int, int]:
    n_rows = len(A)
    n_col = len(A[0]) if A else 0  
    return (n_rows, n_col)

def get_row(A: Matrix, i: int) -> List:
    return A[i]

def get_column(A: Matrix, j: int) -> List:
    return [r[j] for  r in A]


def make_matrix(num_rows: int,
                num_cols: int, 
                entry_fn: Callable[[int, int],float]) -> Matrix:
#Returns a num_rows x num_cols matrix whose (i,j)-th entry is entry_fn(i, j)
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(size: int) -> Matrix:
    return make_matrix(size, size, lambda i, j: 1 if i==j else 0)







