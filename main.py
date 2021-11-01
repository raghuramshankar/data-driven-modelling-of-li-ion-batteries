from plot import plotData
from cellData import cellData

def main():
    filename = "25degC/549_C20DisCh.csv"
    cell = cellData(filename)
    data = cell.extractData()
    plotData(data, filename)

if __name__ == "__main__":
    main()