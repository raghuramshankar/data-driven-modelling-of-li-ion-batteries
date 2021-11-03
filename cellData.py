import numpy as np
import pandas as pd

from funcs import convertToSec


class cellData():
    def __init__(self, filename, pathname):
        self.pathname = pathname
        self.filename = filename
        self.fullname = self.pathname + self.filename
        
    def extractData(self):
        self.df = pd.read_csv(self.fullname, skiprows=28, dtype=str)
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]
        self.df = self.df.drop(0)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")
        self.progTime = [convertToSec(progTime) for progTime in self.df["Prog Time"]]
        self.time = [progTime - self.progTime[0] for progTime in self.progTime]
        self.df["Time"] = [time for time in self.time]

        self.volt = np.asarray([voltage for voltage in self.df["Voltage"]])
        self.curr = np.asarray([- current for current in self.df["Current"]])
        self.disCap = np.asarray([capacity for capacity in self.df["Capacity"]])
        self.dt = np.mean(np.diff(self.time))
        self.eta = 1.0

        print("extract data done from", self.filename)