from plotData import plotData
from cellData import cellData

def main():
    """define desired cell filename"""
    filename = "25degC/549_C20DisCh.csv"

    """define class objects"""
    cell = cellData(filename)
    plot = plotData()

    """run class functions"""
    cell.extractData()
    plot.plotDataFromDataset(cell)

if __name__ == "__main__":
    main()