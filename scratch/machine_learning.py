import random
from typing import TypeVar, List, Tuple
X = TypeVar('X') # Generic type of variable
Y = TypeVar('Y')

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """
    Split data into fractions [prob, 1 - prob]
    """
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return  data[:cut],data[cut:]


def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idx, test_idx = split_data(idxs, 1-test_pct)
    return ([xs[i] for i in train_idx],
            [xs[i] for i in test_idx],
            [ys[i] for i in train_idx],
            [ys[i] for i in test_idx])


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct_prediction = tp+tn
    total = tp+fp+tn+fn
    return correct_prediction/total

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp/ (tp + fn)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)