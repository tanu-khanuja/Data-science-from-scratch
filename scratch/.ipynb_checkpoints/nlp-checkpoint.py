from scratch.linear_algebra import dot, Vector
import math

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    # print(f"{v1=}, {v2=}")
    return dot(v1, v2)/ math.sqrt(dot(v1,v1)*dot(v2,v2))
