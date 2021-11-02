import os

import numpy as np
import pandas as pd

from funcs import convertToSec


class cellDataOCV():
    def __init__(self, filename, pathname):
        self.pathname = pathname
        self.filename = filename
        self.fullname = self.pathname + self.filename
        
    def extractData(self):
        self.df = pd.read_csv(self.fullname, skiprows=28)
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]
        self.df = self.df.drop(0)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")
        # self.df["Time"] = np.linspace(1, len(self.df.index), len(self.df.index))
        # self.time = [time for time in self.df["Time"]]
        # self.stepTime = [float(stepTime[-5:]) for stepTime in self.df["Step Time"]]
        # self.time = [time - self.stepTime[0] for time in self.stepTime]
        # self.stepTime = [time.strptime(stepTime.split(".")[0], "%H:%M:%S") for stepTime in self.df["Step Time"]]
        self.progTime = [convertToSec(progTime) for progTime in self.df["Prog Time"]]
        self.time = [progTime - self.progTime[0] for progTime in self.progTime]
        self.df["Time"] = [time for time in self.time]

        self.volt = [voltage for voltage in self.df["Voltage"]]
        self.curr = [- current for current in self.df["Current"]]
        self.disCap = [capacity for capacity in self.df["Capacity"]]
        self.dt = np.mean(np.diff(self.time))
        self.eta = 1.0

        print("extract done from", self.filename)

    def extractOCV(self):
        self.disOCV = [self.df["Voltage"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "DCH"]
        self.disTime = [self.df["Time"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "DCH"]
        self.chgOCV = np.flip([self.df["Voltage"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "CHA"])
        self.chgTime = [self.df["Time"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "CHA"]
        self.chgTime = self.chgTime - self.chgTime[0]
        self.pauOCV = [self.df["Voltage"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "PAU"]
        self.disCap = [self.df["Capacity"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "DCH"]
        print("OCV done")

    def computeOCV(self):
        self.OCV = (self.disOCV + self.chgOCV[0:len(self.disOCV)])/2
        self.disCapacity = - self.disCap[-1]
        self.SOC = np.flip(np.negative(self.disCap)/self.disCapacity)
        print("compute done")

    def saveOCV(self):
        self.dfOCV = {}
        self.dfOCV.update({"time": self.chgTime[0:len(self.disOCV)]})
        self.dfOCV.update({"OCV": self.OCV})
        self.dfOCV.update({"SOC": self.SOC})
        self.dfOCV.update({"disCapacity": [self.disCapacity for _ in range(len(self.OCV))]})
        self.dfOCV = pd.DataFrame(self.dfOCV)
        self.dfOCV.to_csv("results/OCV--" + self.filename.replace("/", "--"), index=False)
        print("save done")

    def loadOCV(self):
        pathname = "results/"
        filenames = [filename for filename in os.listdir(pathname) if filename.endswith(".csv")]
        index = 0
        self.filename = filenames[index]
        self.dfOCV = pd.read_csv(pathname + self.filename)
        self.timeOCV = self.dfOCV["time"].to_numpy()
        self.voltOCV = self.dfOCV["OCV"].to_numpy()
        self.SOCOCV = self.dfOCV["SOC"].to_numpy()
        self.capacityOCV = self.dfOCV["disCapacity"].to_numpy()[0]
        print("load done")

    def extractDynamic(self):
        self.initSOC = 1.0
        self.testSOC = self.initSOC - self.dt/(self.capacityOCV * 3600) * self.eta * np.cumsum(self.curr)
        # self.testOCV = 
        # self.dynOCV = 
        
        print('dynamic done')
