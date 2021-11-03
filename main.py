import os

import matplotlib.pyplot as plt
import pandas as pd

from cellData import cellData
from cellExtractOCV import cellExtractOCV
from cellExtractParams import cellExtractParams
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

    """extract from dataset"""
    cellDataObj = cellData(filename, pathname)
    plotDataObj = plotData()

    cellDataObj.extractData()

    # plotDataObj.plotDataFromDataset(cellDataObj)

    """extract and save OCV functions"""
    # cellExtractOCVObj = cellExtractOCV(cellDataObj)

    # cellExtractOCVObj.runOCV()

    # plotDataObj.plotComputedOCV(cellExtractOCVObj)

    """extract and save dynamic functions"""
    cellExtractParamsObj = cellExtractParams(cellDataObj)

    cellExtractParamsObj.runSimOpti()

    # plotDataObj.plotLoadedOCV(cellSimObj)
    plotDataObj.plotDynamic(cellExtractParamsObj)
    
    plt.show()

if __name__ == "__main__":
    main()
