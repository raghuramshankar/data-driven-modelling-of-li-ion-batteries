import numpy as np
import pandas as pd

class cellData():
    def __init__(self, filename):
        self.pathname = (
        "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
        )
        self.filename = self.pathname + filename
        
    def extractData(self):
        df = pd.read_csv(self.filename, skiprows=28)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.drop(0)
        df = df.apply(pd.to_numeric, errors="ignore")
        df["Time"] = np.linspace(1, len(df.index), len(df.index))
        print('extract done')
        return df
