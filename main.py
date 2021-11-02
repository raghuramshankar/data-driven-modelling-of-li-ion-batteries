import os

import matplotlib.pyplot as plt
import pandas as pd

from cellData import cellDataOCV
from plotData import plotData


def main():
    """define cell filename"""
    pathname = "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
    temp = "25degC/"
    filenames = [filename for filename in os.listdir(pathname + temp) if filename.endswith(".csv")]
    d = pd.DataFrame(filenames)
    d.to_csv("filenames.csv", header=None, index=False)
    filename = temp + "551_Mixed1.csv"
    # filename = temp + "549_C20DisCh.csv"

    """define class objects"""
    cell = cellDataOCV(filename, pathname)
    plot = plotData()

    """extract from dataset"""
    cell.extractData()

    plot.plotDataFromDataset(cell)

    """extract and save OCV functions"""
    # cell.extractOCV()
    # cell.computeOCV()
    # cell.saveOCV()

    # plot.plotComputedOCV(cell)

    """extract and save dynamic functions"""
    cell.loadOCV()
    cell.extractDynamic()

    plot.plotLoadedOCV(cell)    
    
    plt.show()

if __name__ == "__main__":
    main()
