import os

import matplotlib.pyplot as plt

from cellData import cellData
from plotData import plotData


def main():
    """define cell filename"""
    pathname = "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
    temp = "25degC/"
    filenames = [filename for filename in os.listdir(pathname + temp) if filename.endswith(".csv")]
    filename = temp + filenames[0]

    """define class objects"""
    cell = cellData(filename, pathname)
    plot = plotData()

    """run class functions"""
    cell.extractData()
    cell.extractOCV()
    plot.plotDataFromDataset(cell)
    plot.plotExtractedData(cell)
    plt.show()

if __name__ == "__main__":
    main()
