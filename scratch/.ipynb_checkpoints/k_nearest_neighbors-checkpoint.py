from typing import List
from collections import Counter

def raw_majority_vote(labels: List[str]) -> str:
    """ 
    Outputs label with maximum count
    """
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, winner_votes = votes.most_common(1)[0]
    num_winner = len([count 
                      for count in votes.values()
                      if count == winner_votes])
    if num_winner == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # remove last lable from list

from typing import NamedTuple, List
from scratch.linear_algebra import distance, Vector

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, 
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:
    """ 
    To find label of new_point
    """

    # Order the labeled_points based on distance
    lp_by_distance = sorted(labeled_points, key = lambda lp: distance(new_point, lp.point))

    # Find labels for k- closest
    k_nearest_labels = [lp.label for lp in lp_by_distance[:k]]

    # Let them vote
    return majority_vote(k_nearest_labels)

# Function to parse a row in tuple format [(measurments, class), (measurements, class), ()..]
def parse_iris_row(row: List[str]) -> LabeledPoint:
    """ 
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = [float(value) for value in row[:-1] if len(row)!=0]
    label = row[-1].split('-')[-1] if len(row) !=0 else None
    
    return LabeledPoint(measurements, label)

# read csv file and parse it
import csv
with open('iris.dat') as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader]

# Also group points by class for plotting purpose
from typing import Dict
from collections import defaultdict

points_by_species: Dict[str, List[Vector]] = defaultdict(list)  #default value of a key is empty list
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

from typing import Tuple

# Type how many times we see (predicted,actual)
confusion_matrix: Dict[Tuple[str,str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label

    if predicted == actual:
        num_correct +=1

    confusion_matrix[(predicted, actual)] +=1

import random
from scratch import linear_algebra

random.seed(0)
def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


random.seed(0)
def random_distances(dim: int, num_pairs: int) -> List[float]: 
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]
                            