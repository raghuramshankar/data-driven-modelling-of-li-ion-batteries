# import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("Qt5Agg")

class plotData():
    def plotDataFromDataset(self, cell):
        fig = plt.figure()
        f1 = fig.add_subplot(121)
        f1.plot(cell.df["Time"], cell.df["Voltage"], "b", label="Voltage")
        f1.set_xlabel("Time [s]")
        f1.set_ylabel("Voltage [V]")
        f1.set_title("Voltage from " + cell.filename)
        f1.legend(loc="lower right")
        f1.grid(True)

        f2 = fig.add_subplot(122)
        f2.plot(cell.df["Time"], cell.df["Current"], "b", label="Current")
        f2.set_xlabel("Time [s]")
        f2.set_ylabel("Current [A]")
        f2.set_title("Current from " + cell.filename)
        f2.legend(loc="lower right")
        f2.grid(True)

        plt.show()