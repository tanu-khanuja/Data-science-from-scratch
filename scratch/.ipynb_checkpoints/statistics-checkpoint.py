from typing import List
from scratch.linear_algebra import sum_of_squares
import math
from scratch.linear_algebra import dot

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


#for odd length dataset
# The underscores indicate that these are "private" functions, as they're
# intended to be called by our median function but not by other people
# using our statistics library.
def _median_odd(my_dataset: List[float]) -> float:
    #median is middle element
    return my_dataset[len(my_dataset)//2]

#for even length dataset
def _median_even(my_dataset: List[float]) -> float:
    #median is avg of two middle elements
    low_midpoint = len(my_dataset)//2
    high_midpoint = low_midpoint-1
    return (my_dataset[low_midpoint] + my_dataset[high_midpoint])/2

#Overall median function
def median(my_dataset: List[float]) -> float:
    v = sorted(my_dataset)
    return _median_even(v) if len(v)%2 == 0 else _median_odd(v)


def quantile(my_dataset: List[float], my_quantile: float) -> float:
    my_sorted = sorted(my_dataset)
    return my_sorted[int(len(my_dataset)*my_quantile)]


def mode(my_dataset: List[float]) -> List[float]:
    count_dict = Counter(my_dataset) #create a x:y dict for 'x value is repeting y times'
    max_counts = max(count_dict.values()) #max repeating times
    return [x for x,y in count_dict.items() if y == max_counts] #max repteating key or x

def data_range(my_dataset: List[float]) -> float:
    return max(my_dataset) - min(my_dataset)


#find x-mean, i.e. deviation from mean
def de_mean(my_dataset: List[float]) -> List[float]: 
    my_mean = mean(my_dataset)
    return [(x_i-my_mean) for x_i in my_dataset]
assert de_mean([1,2,3,4,5]) == [-2.0, -1.0, 0.0, 1.0, 2.0]

#find variance = sum of squares of mean deviations/length of dataset

def variance(my_dataset: List[float]) -> float:
    assert len(my_dataset) >=2 #variance requires at least two elements
    l = len(my_dataset)
    return sum_of_squares(de_mean(my_dataset))/l-1

def standard_deviation(my_dataset: List[float]) -> float:
    #SD is square root of variance
    return math.sqrt(variance(my_dataset))


def interquatile_range(my_dataset: List[float]) -> float:
    #values below 75%- values below 25% = values between 25% and 75%
    return quantile(my_dataset,0.75)-quantile(my_dataset,0.25)

def covariance(x: List[float], y: List[float]) -> float:
    assert len(x) == len(y), "different sizes!"
    return dot(de_mean(x),de_mean(y))/(len(x)-1)

def correlation(x: List[float], y:List[float]):
    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    if std_x > 0 and std_y > 0:
        return covariance(x,y)/(standard_deviation(x)*standard_deviation(y))
    else:
        return 0
