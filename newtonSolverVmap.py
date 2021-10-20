import jax as jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import time

jax.config.update('jax_platform_name', 'cpu')

@jit
def func(x): return jnp.sin(x)

@jit
def loss(x): 
    grad = jax.grad(func)
    hess = jax.grad(grad)
    return x - 0.01 * grad(x)/hess(x)

if __name__ == '__main__':
    start = time.time()
    num = 50
    domain = jnp.linspace(-3/2 * jnp.pi, jnp.pi/2, num)
    for _ in range(num):
        domain = jax.vmap(func)(domain)
    min = jax.vmap(loss)(domain)
    arglist = np.nanargmin(min)

    print('arglist = ', arglist)
    print('minimum = ', min[arglist])
    print('argmin in degrees = ', domain[arglist])

    print('Time of execution:', time.time() - start)