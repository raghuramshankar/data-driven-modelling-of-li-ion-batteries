import timeit

import jax.numpy as jnp
from jax import jit, random

size = 5000
key = random.PRNGKey(0)
x = random.normal(key, (size, size), dtype=jnp.float32)
# print(x)


def multi(x):
    return jnp.dot(x, x.T)  # runs on the GPU


multi_jit = jit(multi)


def run():
    # y = multi_jit(x).block_until_ready()
    y = multi(x).block_until_ready()
    print(y)


t = timeit.timeit(stmt=run, number=100)
print(t)
