import jax as jax
from jax import jit
import jax.numpy as jnp
import time
# import matplotlib.pyplot as plt
# import pandas as pd

jax.config.update('jax_platform_name', 'gpu')

@jit
def func(x0):
    # return jnp.sin(x) + jnp.cos(y)
    return jnp.cos(x0)

@jit
def loss(x0, jac = 1, hes = 1):
    jac = jax.grad(func)(x0)
    hes = jax.grad(jax.grad(func))(x0)
    delX = -jac/hes
    x0 = x0 + t * delX
    return jac, x0

if __name__ == '__main__':
    start = time.time()
    x0 = jnp.pi/5
    # y0 = 0
    jac = 1
    # hes = 1
    t = 0.01
    while abs(jac) >= 1e-7:
        jac, x0 = loss(x0)
        # print('jac = ', jac)
        # print('x0 in deg = ', x0 * 180/jnp.pi)

    # print(func(jnp.pi/2, jnp.pi/2))
    # print(jax.grad(func)(jnp.pi/2, jnp.pi/2))

    print('Time of execution:', time.time() - start)