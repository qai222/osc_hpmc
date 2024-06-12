### Problem definition
Given two boxes of shapes, box $A$ and box $B$, define a distance function $D$ such that one can calculate $D(A, B) \in \mathbb{R}^+ \cup {0}$ describing the difference between their packing patterns.
1. There are $m$ identical shapes in box $A$. A shape in $A$ is indexed by $i \in I$ and is entirely defined by a surface $\alpha$. Likewise we use $n$, $j \in J$, $\beta$ for (shapes in) box $B$;
2. The position of $A_i$ is entirely defined by its orientation $o_i$ and the coordinates of its center of mass $c_i$;
3. The orientation of $B$ is $o_B$ defined relative to $A$.

All are defined in 3D unless otherwise specified.

```python
class Box: pass
class Surface: pass
class Orientation: pass
class Coordinates: pass
``` 

### Algorithm 1. Align One
```python
def get_shape_intersection(surface1, surface2) -> float:
    # calculate intersection between two surfaces
    # see https://gist.github.com/qai222/7bd1e561e7a5de9399c78b8b4b64357b for convex shapes
    return intersection

def get_htm_distance(htm) -> float:
    # a way to describe how large the translation + rotation is
    return htm_distance

def get_htm_distances(A, B) -> np.ndarray:
    # a homogeneous transformation matrix is a 4x4 matrix describing both translation and rotation
    # first get a set of matrices indexed by I,J
    # then use `get_htm_distance` to get a value: htms[i, j] -> the value of htm between A_i and B_j
    return htms

def align_boxes_by_ij(i, j):
    # align two boxes such that A_i maximally overlaps with B_j
    return A, B

d = inf
for i in I:
    for j in J:
        # align boxes to perfectly overlap two selected shapes from two boxes
        A, B = align_boxes_by_ij(i, j)

        # get htms for all I J
        htms = get_htm_distances(A, B)

        # use the htms as the cost matrix to get the optimal mapping between I and J
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        row_ind, col_ind = linear_sum_assignment(htms)
        d_ij = htms[row_ind, col_ind].sum()
        if d_ij < d:
            d = d_ij
```
