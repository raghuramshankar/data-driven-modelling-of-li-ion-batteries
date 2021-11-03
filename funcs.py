import numpy as np
from jax import grad, jit, vmap


def convertToSec(progTime):
    [h, m, s] = map(float, progTime.split(":"))
    return h * 3600 + m * 60 + s

def newtonStepOpti(cellExtractParamsObj, alpha, param):
    # simFunc = lambda r0, r = None, c = None: cellExtractParamsObj.cellSim()
    # lossFunc = lambda: cellExtractParamsObj.computeRMS()
    # simFunc(param)
    delX = alpha * grad(cellExtractParamsObj.cellSim)(param, [0], [0])/grad(grad(cellExtractParamsObj.cellSim))(param, [0], [0])
    while delX > 0.001:
        # simFunc(param)
        delX = alpha * grad(cellExtractParamsObj.cellSim)(param, [0], [0])/grad(grad(cellExtractParamsObj.cellSim)(param, [0], [0]))
        param = param - delX
        print("param = ", param)
    return param

def singleGradDescent(cellExtractParamsObj):
    # domain = np.linspace(0.0, 1.0, num=50)
    # vfuncNT = jit(vmap(newtonStep))
    # for _ in range(50):
    #     domain = vfuncNT(domain)

    # minfunc = vmap(cellExtractParamsObj.cellSim)
    # minimums = minfunc(domain)
    # arglist = np.nanargmin(minimums)
    # argmin = domain[arglist]
    # minimum = minimums[arglist]

    initR0 = 0.1
    alpha = 0.1
    r0 = newtonStepOpti(cellExtractParamsObj, alpha, initR0)
    
    print("single opti done")
    return r0

