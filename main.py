from extract import extractData
from plot import plotData

if __name__ == "__main__":
    pathname = (
        "datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/"
    )
    filename = "25degC/549_C20DisCh.csv"
    data = extractData(pathname + filename)
    plotData(data, filename)
