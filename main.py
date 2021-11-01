import os

import matplotlib.pyplot as plt

from cellData import cellDataOCV
from plotData import plotData


def main():
    """define cell filename"""
    pathname = "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
    temp = "25degC/"
    filenames = [filename for filename in os.listdir(pathname + temp) if filename.endswith(".csv")]
    index = 0
    filename = temp + filenames[index]

    """define class objects"""
    cell = cellDataOCV(filename, pathname)
    plot = plotData()

    """extract from dataset"""
    cell.extractData()

    # plot.plotDataFromDataset(cell)

    """run OCV functions"""
    cell.extractOCV()
    cell.computeOCV()
    cell.saveOCV()
    # cell.loadOCV()

    plot.plotComputedOCV(cell)

    """run dynamic functions"""
    # cell.extractDynamic()
    
    
    plt.show()

if __name__ == "__main__":
    main()
