import numpy as np
import pandas as pd

class cellData():
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
        print("extract done")

    # def extractOCV(step):
    #     self.