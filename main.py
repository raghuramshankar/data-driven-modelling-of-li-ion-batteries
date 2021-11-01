from plotData import plotData
from cellData import cellData
import os

def main():
    """define desired cell filename"""
    pathname = "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
    temp = "25degC/"
    filenames = [filename for filename in os.listdir(pathname + temp) if filename.endswith(".csv")]
    filename = temp + filenames[0]

    """define class objects"""
    cell = cellData(filename, pathname)
    plot = plotData()

    """run class functions"""
    cell.extractData()
    plot.plotDataFromDataset(cell)

if __name__ == "__main__":
    main()