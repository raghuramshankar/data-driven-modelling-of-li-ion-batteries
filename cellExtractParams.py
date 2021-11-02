import os

import numpy as np
import pandas as pd


class cellExtractParams():
    def __init__(self, cellDataObj):
        self.filename = cellDataObj.filename
        self.time = cellDataObj.time
        self.volt = cellDataObj.volt
        self.curr = cellDataObj.curr
        self.dt = cellDataObj.dt
        self.eta = cellDataObj.eta
        self.nTime = len(cellDataObj.time)

    def loadOCV(self):
        pathname = "results/"
        filenames = [filename for filename in os.listdir(pathname) if filename.endswith(".csv")]
        index = 0
        self.filenameOCV = filenames[index]
        self.dfOCV = pd.read_csv(pathname + self.filenameOCV)
        self.timeOCV = self.dfOCV["time"].to_numpy()
        self.voltOCV = self.dfOCV["OCV"].to_numpy()
        self.SOCOCV = self.dfOCV["SOC"].to_numpy()
        self.capacityOCV = self.dfOCV["disCapacity"].to_numpy()[0]

        print("load OCV done")

    def loadCellParams(self):
        self.r0 = 1e-3
        self.r = [1e-3, 3e-3]
        self.c = [2e3, 5e3]
        self.nRC = len(self.r)

    def extractDynamic(self):
        self.initSOC = self.SOCOCV[np.argmin(abs(self.voltOCV - self.volt[0]))]
        self.testSOC = self.initSOC - self.dt/(self.capacityOCV * 3600) * self.eta * np.cumsum(self.curr)
        self.testOCV = [self.voltOCV[np.argmin(abs(self.SOCOCV - soc))] for soc in self.testSOC]
        self.overPotVolt = self.volt - self.testOCV
        
        print('extract dynamic done')

    def cellSim(self):
        self.iR = np.zeros((self.nTime, self.nRC))
        self.vC = np.zeros((self.nTime, self.nRC))
        self.f = np.diag([np.exp(-self.dt/(self.r[j] * self.c[j])) for j in range(len(self.r))])
        

        print('cell sim done')
