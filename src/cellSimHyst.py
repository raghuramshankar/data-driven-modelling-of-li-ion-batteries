import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class cellSimHyst:
    def __init__(self, cellDataObj):
        self.filename = cellDataObj.filename
        self.time = cellDataObj.time
        self.volt = cellDataObj.volt
        self.curr = cellDataObj.curr
        self.dt = cellDataObj.dt
        self.eta = cellDataObj.eta
        self.nRC = 2
        self.nTime = len(cellDataObj.time)
        self.volt = self.volt[0: self.nTime]
        self.s = np.zeros_like(self.curr)

    def loadOCV(self):
        pathname = "results/"
        filenames = [
            filename for filename in os.listdir(pathname) if filename.startswith("OCV")
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
        for i in range(self.nTime):
            if np.abs(self.curr[i]) > 0:
                self.s[i] = np.sign(self.curr[i])
            else:
                self.s[i] = self.s[i-1]

        print("extract dynamic done")

    def loadCellParamsOpti(self):
        pathname = "results/"
        filenames = [
            filename for filename in os.listdir(pathname) if filename.startswith("CellParamsHyst")
        ]
        index = 0
        self.filenameCellParamsOpti = filenames[index]
        self.dfCellParamsHystOpti = pd.read_csv(
            pathname + self.filenameCellParamsOpti)
        self.r0 = self.dfCellParamsHystOpti["r0"].to_numpy()
        self.r1 = self.dfCellParamsHystOpti["r1"].to_numpy()
        self.r2 = self.dfCellParamsHystOpti["r2"].to_numpy()
        self.c1 = self.dfCellParamsHystOpti["c1"].to_numpy()
        self.c2 = self.dfCellParamsHystOpti["c2"].to_numpy()
        self.m0 = self.dfCellParamsHystOpti["m0"].to_numpy()
        self.m = self.dfCellParamsHystOpti["m"].to_numpy()
        self.gamma = self.dfCellParamsHystOpti["gamma"].to_numpy()
        # self.m0 = 1.0
        # self.m = 1.0

        print("load CellParamsOpti done from " + self.filenameCellParamsOpti)

    def computeRMS(self):
        self.rmsError = 1000 * np.sqrt(np.mean(np.square(self.vT - self.volt)))

        return self.rmsError

    def cellSimHyst(self):
        self.iR = np.zeros((self.nRC, self.nTime))
        self.vC = np.zeros((self.nRC, self.nTime))
        self.h = np.zeros(self.nTime)
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
            self.aH = np.exp(-abs(self.eta *
                             self.curr[k] * self.gamma * self.dt/self.capacityOCV))
            self.h[k + 1] = self.aH * self.h[k] - \
                (1 - self.aH) * np.sign(self.curr[k])
            self.vT[k + 1] = (
                self.testOCV[k] - np.sum(self.vC[:, k]) -
                self.curr[k] * self.r0 + self.m0 *
                self.s[k] + self.m * self.h[k]
            )

    def printCellParams(self):
        print("R0 = ", self.r0, "ohm")
        print("R1 = ", self.r1, "ohm")
        print("R2 = ", self.r2, "ohm")
        print("C1 = ", self.c1, "farad")
        print("C2 = ", self.c2, "farad")
        print("M0 = ", self.m0)
        print("M = ", self.m)
        print("Gamma = ", self.gamma)

    def objFn(self, x0):
        self.r0 = x0[3]
        self.r1 = x0[4]
        self.r2 = x0[5]
        self.c1 = x0[6] * self.scaleFactorC
        self.c2 = x0[7] * self.scaleFactorC
        self.m0 = x0[0] * self.scaleFactorM
        self.m = x0[1] * self.scaleFactorM
        self.gamma = x0[2] * self.scaleFactorM
        # self.m = self.m
        # self.gamma = self.gamma
        self.cellSimHyst()
        rmsError = self.computeRMS()
        return rmsError

    def constraintR0(self, x):
        return x[0]

    def constraintRC1(self, x):
        return 1 - x[1] * x[3]

    def constraintRC2(self, x):
        return 10 - x[2] * x[4]

    def optFn(self):
        print("started hysteresis parameter extraction via optimization")
        self.scaleFactorC = 1e6
        self.scaleFactorM = 1e3
        x0 = [100.0/self.scaleFactorM, 100.0/self.scaleFactorM, 100.0/self.scaleFactorM, 1e-3, 1e-2, 1e-2, 100e3 /
              self.scaleFactorC, 100e3/self.scaleFactorC]
        bndsM = (0/self.scaleFactorM, 1e3/self.scaleFactorM)
        bndsR0 = (0.1e-3, 100e-3)
        bndsR = (1e-3, 5000e-3)
        bndsC = (1e3/self.scaleFactorC, 500e3/self.scaleFactorC)
        bnds = (bndsM, bndsM, bndsM, bndsR0, bndsR, bndsR, bndsC, bndsC)
        # constraint1 = {"type": "ineq", "fun": self.constraintR0}
        # constraint2 = {"type": "ineq", "fun": self.constraintRC1}
        # constraint3 = {"type": "ineq", "fun": self.constraintRC2}
        # cons = [constraint1, constraint2, constraint3]
        # minimize(self.objFn, x0, method="SLSQP", bounds=bnds, constraints=cons)
        minimize(self.objFn, x0, method="SLSQP", bounds=bnds)

    def saveCellParamsOpti(self):
        self.filenameCellParamsHystOpti = "results/CellParamsHyst--" + \
            self.filename.replace("/", "--")
        self.dfCellParamsHyst = {}
        self.dfCellParamsHyst.update({"r0": self.r0})
        self.dfCellParamsHyst.update({"r1": self.r1})
        self.dfCellParamsHyst.update({"r2": self.r2})
        self.dfCellParamsHyst.update({"c1": self.c1})
        self.dfCellParamsHyst.update({"c2": self.c2})
        self.dfCellParamsHyst.update({"m0": self.m0})
        self.dfCellParamsHyst.update({"m": self.m})
        self.dfCellParamsHyst.update({"gamma": self.gamma})
        self.dfCellParamsHyst = pd.DataFrame(self.dfCellParamsHyst, index=[0])
        self.dfCellParamsHyst.to_csv(
            self.filenameCellParamsHystOpti, index=False)

    def runSimValidate(self):
        print("starting validation of RC2 hysteresis cell model")
        self.loadOCV()
        self.extractDynamic()
        self.loadCellParamsOpti()
        self.printCellParams()
        self.cellSimHyst()
        print("CRMSE = ", self.computeRMS())

    def runSimTrain(self):
        print("starting training of RC2 hysteresis cell model")
        self.loadOCV()
        self.loadCellParamsOpti()
        self.extractDynamic()
        self.optFn()
        self.saveCellParamsOpti()
        self.printCellParams()
        self.cellSimHyst()
        print("CRMSE = ", self.computeRMS())
