import numpy as np
import pandas as pd


def extractData(filename):
    df = pd.read_csv(filename, skiprows=28)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.drop(0)
    df = df.apply(pd.to_numeric, errors="ignore")
    df["Time"] = np.linspace(1, len(df.index), len(df.index))
    return df
