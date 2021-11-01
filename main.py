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

    """run OCV functions"""
    cell.extractData()
    cell.extractOCV()
    cell.computeOCV()
    cell.saveOCV()

    """run dynamic functions"""
    cell.extractDynamic()

    """run plotting functions"""
    # plot.plotDataFromDataset(cell)
    plot.plotComputedOCV(cell)
    
    plt.show()

if __name__ == "__main__":
    main()
