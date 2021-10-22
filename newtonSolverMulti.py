import time

import jax as jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_platform_name", "gpu")


@jit
def func(x, y):
    return x * x + y * y - 100
    # return jnp.cos(x0)
    # return jnp.sqrt(jnp.tan(x0) * jnp.square(x0)/jnp.cos(x0*jnp.sqrt(x0)))
    # return jnp.sqrt(100 - jnp.square(x0)) + 0


@jit
def loss(x):
    jac = jax.jacfwd(func)()
    hes = jax.grad(jax.grad(func))(x0)
    delX = -jac / hes
    x0 = x0 + t * delX
    y = func(x0)
    return jac, x0, y


if __name__ == "__main__":
    start = time.time()
    # x0 = jnp.pi/5
    x0 = 9.0
    # y0 = 0
    jac = 1.0
    # hes = 1
    t = 1.0
    for _ in range(5):
        jac, x0, y = loss(x0)
        print("jac = ", jac)
        # print('x0 in deg = ', x0 * 180/jnp.pi)
        print("x0 = ", x0)
        print("y = ", y)
        print("\n")

    # print(func(jnp.pi/2, jnp.pi/2))
    # print(jax.grad(func)(jnp.pi/2, jnp.pi/2))

    print("Time of execution:", time.time() - start)
