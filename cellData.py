import numpy as np
import pandas as pd

class cellData():
    def __init__(self, filename):
        self.pathname = (
        "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
        )
        self.filename = self.pathname + filename
        
    def extractData(self):
        self.df = pd.read_csv(self.filename, skiprows=28)
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]
        self.df = self.df.drop(0)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")
        self.df["Time"] = np.linspace(1, len(self.df.index), len(self.df.index))
        print('extract done')
