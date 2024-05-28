from scratch.linear_algebra import Vector, dot
import math
from typing import List
from typing import List
Vector = List[float]

# Threshold 

def step_function(x: float) -> float:
    return 1.0 if x>=0 else 0.0

assert step_function(-1) == 0
assert step_function(1) == 1
assert step_function(0) == 1

# Activation Function


def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    weighted_sum = dot(weights, x) + bias
    return step_function(weighted_sum)

# Sigmoid Function


def sigmoid(t: float) -> float:
    return 1/(1+math.exp(-t))

# Neuron output
def neuron_output(weights: Vector, inputs: Vector) -> float:
    # Weights include bias term and inputs include a 1
    return sigmoid(dot(weights, inputs))

# Feed Forward


def feed_forward(neural_netwrok: List[List[Vector]], 
                 input_vector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """
    outputs: List[Vector] = []

    for layer in neural_netwrok:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output

    return outputs

# Backpropagation

# 1. Compute Gradients

def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector, and a target vector,
    make a prediction and compute the gradient of the squared error
    loss with respect to the neuron weights.
    """
    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients wrt output neuron pre-actiavtion outputs
    output_deltas = [output * (1 - output)* (output - target)
                     for output, target in zip(outputs, target_vector)]
    
    output_grads = [[output_deltas[i]*hidden_output 
                     for hidden_output in hidden_outputs +[1]] 
                    for i, output_neuron in enumerate(network[-1])]

    # gradients wrt hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) * dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input
                     for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]
    
    return [hidden_grads, output_grads]

# FizzBuzz
# Input encoding

def binary_encode(x: int) -> Vector:
    binary: List[float] = []
    for i in range(10):
        binary.append(x%2)
        x = x//2 # Divide
    return binary
# output decoding

def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def argmax(xs: list) -> int:
    """Returns the index of the largest value"""
    return max(range(len(xs)), key=lambda i: xs[i])


    return binary