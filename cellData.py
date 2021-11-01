import os

import numpy as np
import pandas as pd


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
        self.df["Time"] = np.linspace(1, len(self.df.index), len(self.df.index))
        self.time = [time for time in self.df["Time"]]
        self.voltage = [voltage for voltage in self.df["Voltage"]]
        self.current = [current for current in self.df["Current"]]
        print("extract done")

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
        self.capacity = self.disCap[-1]
        self.SOC = self.disCap/self.capacity
        print("compute done")

    def saveOCV(self):
        self.dfOCV = {}
        self.dfOCV.update({"time": self.chgTime[0:len(self.disOCV)]})
        self.dfOCV.update({"OCV": self.OCV})
        self.dfOCV.update({"SOC": self.SOC})
        self.dfOCV = pd.DataFrame(self.dfOCV)
        self.dfOCV.to_csv("results/OCV--" + self.filename.replace("/", "--"), index=False)
        print("save done")

    def loadOCV(self):
        pathname = "results/"
        filenames = [filename for filename in os.listdir(pathname) if filename.endswith(".csv")]
        index = 0
        self.dfOCV = pd.read_csv(pathname + filenames[index])
        self.time = self.dfOCV["time"]
        self.OCV = self.dfOCV["OCV"]
        print("load done")

    def extractDynamic(self):
        # self.dynOCV = 
        pass
