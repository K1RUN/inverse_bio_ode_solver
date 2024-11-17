import numpy as np
import timeit
import jax


@jax.jit
def pairwise_distances_jit(x):
    # Using the identity \|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 x^T y
    distances = (
        np.sum(x**2, axis=1)[:, None] - 2 * x @ x.T + np.sum(x**2, axis=1)[None, :]
    )

    return distances


def pairwise_distances(x):
    # Using the identity \|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 x^T y
    distances = (
        np.sum(x**2, axis=1)[:, None] - 2 * x @ x.T + np.sum(x**2, axis=1)[None, :]
    )

    return distances


def get_sq_dist(vector1, vector2):
    diff = (vector1[:, np.newaxis] - vector2)**2
    return np.sum(diff, axis=-1)


@jax.jit
def get_sq_dist_jit(vector1, vector2):
    sq_diff = (vector1[:, np.newaxis] - vector2) ** 2
    return np.sum(sq_diff, axis=-1)


def pairwise_distance_matrix(vectors1, vectors2):
    # Ensure vectors are n-dimensional arrays
    vectors1 = np.atleast_1d(vectors1)
    vectors2 = np.atleast_1d(vectors2)

    # Calculate differences between all pairs of vectors
    diff = vectors1[:, np.newaxis] - vectors2

    # Square the differences
    squared_diff = diff ** 2

    # Sum along the last axis to get squared distances
    sum_squared_diff = np.sum(squared_diff, axis=-1)

    # Take the square root to get the final distances
    distances = np.sqrt(sum_squared_diff)

    return distances


def ise_kernel(a, b):
    sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return sq_dist


n_dimensions = 5
n_points = 350
seed = np.random.randint(0, 10000)
key = jax.random.PRNGKey(seed)

# Generate sample data
x_train = np.linspace(-50, 50, n_points).reshape(-1, 1)
vectors1 = np.random.rand(n_points, n_dimensions)
# dist1 = ise_kernel(x_train, x_train)
# dist2 = get_sq_dist(x_train, x_train)
# dist3 = get_sq_dist(vectors1, vectors1)
# distances = pairwise_distances(x_train)
dist = get_sq_dist_jit(vectors1, vectors1[:200])
dist = np.array(dist)
dist2 = ise_kernel(vectors1, vectors1[:200])
dist3 = pairwise_distances(vectors1)
print()
print("JIT: ", timeit.timeit("get_sq_dist_jit(x_train, x_train)", number=1, globals=globals()))
# print("NO JIT: ", timeit.timeit("get_sq_dist(x_train, x_train)", number=10000, globals=globals()))