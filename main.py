import os

import matplotlib.pyplot as plt
import pandas as pd

from src.cellData import cellData
from src.cellExtractOCV import cellExtractOCV
from src.cellSim import cellSim
from src.cellSimHyst import cellSimHyst
from src.plotData import plotData


def main():
    """create list of tests available from dataset"""
    pathname = (
        "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
    )
    '''define test temperature in deg C'''
    temp = "25degC/"
    '''obtain a list of filenames in above pathname'''
    filenames = [
        filename
        for filename in os.listdir(pathname + temp)
        if filename.endswith(".csv")
    ]
    '''create pandas dataframe with length of filenames as number of rows'''
    d = pd.DataFrame(filenames)
    '''convert dataframe to csv'''
    d.to_csv("filenames.csv", header=None, index=False)
    '''save dataframe as list of available tests done to csv'''
    filename = temp + "549_C20DisCh.csv"

    '''create class objects'''
    cellDataObj = cellData(filename, pathname)
    plotDataObj = plotData()

    cellDataObj.extractData()
    # plotDataObj.plotDataFromDataset(cellDataObj)

    # cellExtractOCVObj = cellExtractOCV(cellDataObj)
    # cellExtractOCVObj.runOCV()
    # plotDataObj.plotComputedOCV(cellExtractOCVObj)

    """train/validate model parameters"""
    # filename = temp + "551_Mixed1.csv"
    # filename = temp + "551_US06.csv"
    filename = temp + "551_LA92.csv"
    cellDataObj = cellData(filename, pathname)

    cellDataObj.extractData()
    # plotDataObj.plotDataFromDataset(cellDataObj)

    # cellSimObj = cellSim(cellDataObj)
    # cellSimObj.runSimTrain()
    # cellSimObj.runSimValidate()
    # plotDataObj.plotLoadedOCV(cellSimObj)
    # plotDataObj.plotDynamic(cellSimObj)

    cellSimHystObj = cellSimHyst(cellDataObj)
    # cellSimHystObj.runSimTrain()
    cellSimHystObj.runSimValidate()
    # plotDataObj.plotLoadedOCV(cellSimHystObj)
    plotDataObj.plotDynamic(cellSimHystObj)

    plt.show()


if __name__ == "__main__":
    main()
