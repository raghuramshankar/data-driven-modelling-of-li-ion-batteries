import matplotlib
from matplotlib import figure
import matplotlib.pyplot as plt

# matplotlib.use("Qt5Agg")
matplotlib.rcParams['figure.dpi'] = 150

class plotData():
    def plotDataFromDataset(self, cell):
        fig = plt.figure(figsize=(10.0, 5.0))
        f1 = fig.add_subplot(121)
        f1.plot(cell.df["Time"], cell.df["Voltage"], "b", label="Voltage")
        f1.set_xlabel("Time [s]")
        f1.set_ylabel("Voltage [V]")
        f1.set_title("Voltage from \n" + cell.filename)
        f1.legend(loc="lower right")
        f1.grid(True)

        f2 = fig.add_subplot(122)
        f2.plot(cell.df["Time"], cell.df["Current"], "b", label="Current")
        f2.set_xlabel("Time [s]")
        f2.set_ylabel("Current [A]")
        f2.set_title("Current from \n" + cell.filename)
        f2.legend(loc="lower right")
        f2.grid(True)

        plt.show()
