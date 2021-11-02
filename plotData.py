import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("Qt5Agg")
matplotlib.rcParams['figure.dpi'] = 150

class plotData():
    def __init__(self):
        self.l = 15.0
        self.h = 5.0

    def plotDataFromDataset(self, cell):
        fig1 = plt.figure(figsize=(self.l, self.h))
        fig1_f1 = fig1.add_subplot(121)
        fig1_f1.plot(cell.time, cell.volt, "b", label="Voltage")
        fig1_f1.set_xlabel("Time [s]")
        fig1_f1.set_ylabel("Voltage [V]")
        fig1_f1.set_title("Voltage from \n" + cell.filename)
        fig1_f1.legend(loc="lower right")
        fig1_f1.grid(True)

        fig1_f2 = fig1.add_subplot(122)
        fig1_f2.plot(cell.time, cell.curr, "b", label="Current")
        fig1_f2.set_xlabel("Time [s]")
        fig1_f2.set_ylabel("Current [A]")
        fig1_f2.set_title("Current from \n" + cell.filename)
        fig1_f2.legend(loc="lower right")
        fig1_f2.grid(True)

    def plotComputedOCV(self, cell):
        fig2 = plt.figure(figsize=(self.l, self.h))
        fig2_f1 = fig2.add_subplot(141)
        fig2_f1.plot(cell.disTime, cell.disOCV, "r", label="Discharge OCV")
        fig2_f1.set_xlabel("Time [s]")
        fig2_f1.set_ylabel("Voltage [V]")
        fig2_f1.set_title("Discharge OCV from \n" + cell.filename)
        fig2_f1.legend()
        fig2_f1.grid(True)

        fig2_f2 = fig2.add_subplot(142)
        fig2_f2.plot(cell.chgTime, cell.chgOCV, "g", label="Charge OCV")
        fig2_f2.set_xlabel("Time [s]")
        fig2_f2.set_ylabel("Voltage [V]")
        fig2_f2.set_title("Charge OCV from \n" + cell.filename)
        fig2_f2.legend()
        fig2_f2.grid(True)

        fig2_f3 = fig2.add_subplot(143)
        fig2_f3.plot(cell.disTime, cell.disOCV, "r--", label="Discharge OCV")
        fig2_f3.plot(cell.chgTime, cell.chgOCV, "g--", label="Charge OCV")
        fig2_f3.plot(cell.disTime, cell.OCV, "b", label="Average OCV")
        fig2_f3.set_xlabel("Time [s]")
        fig2_f3.set_ylabel("Voltage [V]")
        fig2_f3.set_title("Average OCV from \n" + cell.filename)
        fig2_f3.legend()
        fig2_f3.grid(True)

        fig2_f4 = fig2.add_subplot(144)
        fig2_f4.plot(cell.SOC, cell.OCV, "b", label="Average OCV")
        fig2_f4.set_xlabel("SOC [%]")
        fig2_f4.set_ylabel("Voltage [V]")
        fig2_f4.set_title("Average OCV from \n" + cell.filename)
        fig2_f4.legend()
        fig2_f4.grid(True)
    
    def plotLoadedOCV(self, cell):
        fig3 = plt.figure(figsize=(self.l, self.h))
        fig3_f1 = fig3.add_subplot(111)
        fig3_f1.plot(cell.SOCOCV, cell.voltOCV, "b", label="Average OCV")
        fig3_f1.set_xlabel("SOC [%]")
        fig3_f1.set_ylabel("Voltage [V]")
        fig3_f1.set_title("Average OCV from \n" + cell.filename)
        fig3_f1.legend()
        fig3_f1.grid(True)

    def plotDynVolt(self, cell):
        fig4 = plt.figure(figsize=(self.l, self.h))
        fig4_f1 = fig4.add_subplot(121)
        fig4_f1.plot(cell.time, cell.testOCV, "b", label="Test OCV")
        fig4_f1.set_xlabel("Time [sec]")
        fig4_f1.set_ylabel("Test OCV [V]")
        fig4_f1.set_title("Test OCV from \n" + cell.filename)
        fig4_f1.legend()
        fig4_f1.grid(True)

        fig4_f2 = fig4.add_subplot(122)
        fig4_f2.plot(cell.time, cell.overPotVolt, "b", label="Overpotential")
        fig4_f2.set_xlabel("Time [sec]")
        fig4_f2.set_ylabel("Overpotential [V]")
        fig4_f2.set_title("Overpotential from \n" + cell.filename)
        fig4_f2.legend()
        fig4_f2.grid(True)