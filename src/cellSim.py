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
        self.volt = self.volt[0: self.nTime]
        self.sign = np.zeros_like(self.curr)

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
        print("extract dynamic done")

    def loadCellParamsOpti(self):
        pathname = "results/"
        filenames = [
            filename for filename in os.listdir(pathname) if filename.startswith("CellParams")
        ]
        index = 1
        self.filenameCellParamsOpti = filenames[index]
        self.dfCellParamsOpti = pd.read_csv(
            pathname + self.filenameCellParamsOpti)
        self.r0 = self.dfCellParamsOpti["r0"].to_numpy()
        self.r1 = self.dfCellParamsOpti["r1"].to_numpy()
        self.r2 = self.dfCellParamsOpti["r2"].to_numpy()
        self.c1 = self.dfCellParamsOpti["c1"].to_numpy()
        self.c2 = self.dfCellParamsOpti["c2"].to_numpy()

        print("load CellParamsOpti done from " + self.filenameCellParamsOpti)

    def computeRMS(self):
        self.rmsError = 1000 * np.sqrt(np.mean(np.square(self.vT - self.volt)))

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
                self.testOCV[k] - np.sum(self.vC[:, k]) -
                self.curr[k] * self.r0
            )

    def printCellParams(self):
        print("R0 = ", self.r0, "ohm")
        print("R1 = ", self.r1, "ohm")
        print("R2 = ", self.r2, "ohm")
        print("C1 = ", self.c1, "farad")
        print("C2 = ", self.c2, "farad")

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
        x0 = [1e-3, 1e-2, 1e-2, 100e3 /
              self.scaleFactorC, 100e3/self.scaleFactorC]
        bndsR0 = (0.1e-3, 50e-3)
        bndsR = (1e-3, 5000e-3)
        bndsC1 = (1e3/self.scaleFactorC, 500e3/self.scaleFactorC)
        bndsC2 = (1e3/self.scaleFactorC, 5000e3/self.scaleFactorC)
        bnds = (bndsR0, bndsR, bndsR, bndsC1, bndsC2)
        # constraint1 = {"type": "ineq", "fun": self.constraintR0}
        # constraint2 = {"type": "ineq", "fun": self.constraintRC1}
        # constraint3 = {"type": "ineq", "fun": self.constraintRC2}
        # cons = [constraint1, constraint2, constraint3]
        # minimize(self.objFn, x0, method="SLSQP", bounds=bnds, constraints=cons)
        minimize(self.objFn, x0, method="SLSQP", bounds=bnds)

    def saveCellParamsOpti(self):
        self.filenameCellParamsOpti = "results/CellParams--" + \
            self.filename.replace("/", "--")
        self.dfCellParams = {}
        self.dfCellParams.update({"r0": self.r0})
        self.dfCellParams.update({"r1": self.r1})
        self.dfCellParams.update({"r2": self.r2})
        self.dfCellParams.update({"c1": self.c1})
        self.dfCellParams.update({"c2": self.c2})
        self.dfCellParams = pd.DataFrame(self.dfCellParams, index=[0])
        self.dfCellParams.to_csv(self.filenameCellParamsOpti, index=False)

    def runSimValidate(self):
        print("starting validation of RC2 cell model")
        self.loadOCV()
        self.extractDynamic()
        self.loadCellParamsOpti()
        self.printCellParams()
        self.cellSim()
        print("CRMSE = ", self.computeRMS(), "mV")

    def runSimTrain(self):
        print("starting training of RC2 cell model")
        self.loadOCV()
        self.extractDynamic()
        self.optFn()
        self.saveCellParamsOpti()
        self.printCellParams()
        self.cellSim()
        print("CRMSE = ", self.computeRMS(), "mV")
