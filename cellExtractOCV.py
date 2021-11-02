import numpy as np
import pandas as pd


class cellExtractOCV():
    def __init__(self, cellDataObj):
        self.df = cellDataObj.df
        self.filename = cellDataObj.filename
        
    def extractOCV(self):
        self.disOCV = [self.df["Voltage"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "DCH"]
        self.disTime = [self.df["Time"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "DCH"]
        self.chgOCV = np.flip([self.df["Voltage"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "CHA"])
        self.chgTime = [self.df["Time"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "CHA"]
        self.chgTime = self.chgTime - self.chgTime[0]
        self.pauOCV = [self.df["Voltage"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "PAU"]
        self.disCap = [self.df["Capacity"].to_numpy()[i] for i in range(len(self.df)) if self.df["Status"].to_numpy()[i] == "DCH"]
        print("extract OCV done")

    def computeOCV(self):
        self.OCV = (self.disOCV + self.chgOCV[0:len(self.disOCV)])/2
        self.disCapacity = - self.disCap[-1]
        self.SOC = np.flip(np.negative(self.disCap)/self.disCapacity)
        print("compute OCV done")

    def saveOCV(self):
        self.dfOCV = {}
        self.dfOCV.update({"time": self.chgTime[0:len(self.disOCV)]})
        self.dfOCV.update({"OCV": self.OCV})
        self.dfOCV.update({"SOC": self.SOC})
        self.dfOCV.update({"disCapacity": [self.disCapacity for _ in range(len(self.OCV))]})
        self.dfOCV = pd.DataFrame(self.dfOCV)
        self.dfOCV.to_csv("results/OCV--" + self.filename.replace("/", "--"), index=False)
        print("save OCV done")

    def OCV(self):
        self.extractOCV()
        self.computeOCV()
        self.saveOCV()
