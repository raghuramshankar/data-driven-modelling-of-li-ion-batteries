import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class cellSim:
    def __init__(self, cellDataObj):
        self.filename = cellDataObj.filename
        self.time = cellDataObj.time
        self.volt = cellDataObj.volt
        self.curr = cellDataObj.curr
        self.dt = cellDataObj.dt
        self.eta = cellDataObj.eta
        self.nRC = 2
        self.nTime = len(cellDataObj.time)
        # self.nTime = 100
        self.volt = self.volt[0 : self.nTime]

    def loadOCV(self):
        pathname = "results/"
        filenames = [
            filename for filename in os.listdir(pathname) if filename.endswith(".csv")
        ]
        index = 0
        self.filenameOCV = filenames[index]
        self.dfOCV = pd.read_csv(pathname + self.filenameOCV)
        self.timeOCV = self.dfOCV["time"].to_numpy()
        self.voltOCV = self.dfOCV["OCV"].to_numpy()
        self.SOCOCV = self.dfOCV["SOC"].to_numpy()
        self.capacityOCV = self.dfOCV["disCapacity"].to_numpy()[0]

        print("load OCV done")

    def extractDynamic(self):
        self.initSOC = self.SOCOCV[np.argmin(abs(self.voltOCV - self.volt[0]))]
        self.testSOC = self.initSOC - self.dt / (
            self.capacityOCV * 3600
        ) * self.eta * np.cumsum(self.curr)
        self.testOCV = [
            self.voltOCV[np.argmin(abs(self.SOCOCV - soc))] for soc in self.testSOC
        ]
        self.overPotVolt = self.volt - self.testOCV

        print("extract dynamic done")

    def loadCellParamsDefault(self):
        self.r0 = 1e-3
        self.r1 = 50e-3
        self.c1 = 1e3
        self.r2 = 50e-3
        self.c2 = 1e3

        print("load cell params done")

    def computeRMS(self):
        self.rmsError = 1000 * np.sqrt(np.mean(np.square(self.vT - self.volt)))

        # print("RMS error = ", self.rmsError)

        return self.rmsError

    def cellSim(self):
        self.iR = np.zeros((self.nRC, self.nTime))
        self.vC = np.zeros((self.nRC, self.nTime))
        self.vT = np.zeros(self.nTime)
        self.vT[0] = self.testOCV[0]
        self.f = [
            np.exp(-self.dt / np.dot(self.r1, self.c1)),
            np.exp(-self.dt / np.dot(self.r2, self.c2))
        ]
        self.aRC = np.diag(self.f)
        self.bRC = np.ones(self.nRC) - self.f
        for k in range(self.nTime - 1):
            self.iR[:, k + 1] = (
                np.dot(self.aRC, self.iR[:, k]) + self.bRC * self.curr[k]
            )
            self.vC[0, k] = self.iR[0, k] * self.r1
            self.vC[1, k] = self.iR[1, k] * self.r2
            self.vT[k + 1] = (
                self.testOCV[k] - np.sum(self.vC[:, k]) - self.curr[k] * self.r0
            )

        # print("cell sim done")

    def objFn(self, x0):
        self.r0 = x0[0]
        self.r1 = x0[1]
        self.r2 = x0[2]
        self.c1 = x0[3] * self.scaleFactorC
        self.c2 = x0[4] * self.scaleFactorC
        self.cellSim()
        rmsError = self.computeRMS()
        return rmsError
    
    def constraintR0(self, x):
        return x[0]
    
    def constraintRC1(self, x):
        return 1 - x[1] * x[3]

    def constraintRC2(self, x):
        return 10 - x[2] * x[4]

    def optFn(self):
        print("started parameter extraction via optimization")
        self.scaleFactorC = 1e3
        x0 = [10e-3, 50e-3, 100e-3, 100e3/self.scaleFactorC, 200e3/self.scaleFactorC]
        bndsR0 = (1e-3, 50e-3)
        bndsR = (10e-3, 500e-3)
        bndsC1 = (1000/self.scaleFactorC, 20000/self.scaleFactorC)
        bndsC2 = (10000/self.scaleFactorC, 100000/self.scaleFactorC)
        bnds = (bndsR, bndsR, bndsR, bndsC1, bndsC2)
        constraint1 = {"type": "ineq", "fun": self.constraintR0}
        constraint2 = {"type": "ineq", "fun": self.constraintRC1}
        constraint3 = {"type": "ineq", "fun": self.constraintRC2}
        cons = [constraint1, constraint2, constraint3]
        # minimize(self.objFn, x0, method="SLSQP", bounds=bnds, constraints=cons)
        minimize(self.objFn, x0, method="SLSQP", bounds=bnds)
        print("R0 = ", self.r0)
        print("R1 = ", self.r1)
        print("R2 = ", self.r2)
        print("C1 = ", self.c1)
        print("C2 = ", self.c2)
        print("RMS error = ", self.computeRMS())
        

    def runSimLoad(self):
        self.loadOCV()
        self.extractDynamic()
        self.optFn()
        # self.loadCellParamsDefault()
        self.cellSim()
