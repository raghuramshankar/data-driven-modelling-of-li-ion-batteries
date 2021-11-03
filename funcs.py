import numpy as np
from jax import grad, jit, vmap


def convertToSec(progTime):
    [h, m, s] = map(float, progTime.split(":"))
    return h * 3600 + m * 60 + s

def newtonStep(x, alpha):
    return x - alpha * grad(x)/grad(grad(x))

def singleOpti(cellExtractParamsObj):
    domain = np.linspace(3.0, 5.0, num=50)
    vfuncNT = jit(vmap(newtonStep))
    for _ in range(50):
        domain = vfuncNT(domain)

    minfunc = vmap(cellExtractParamsObj.cellSim)
    minimums = minfunc(domain)
    arglist = np.nanargmin(minimums)
    argmin = domain[arglist]
    minimum = minimums[arglist]
    
    return minimum, argmin