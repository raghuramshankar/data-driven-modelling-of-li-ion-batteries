import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 100

class plotData():
    def __init__(self):
        self.l = 15.0
        self.h = 5.0

    def plotDataFromDataset(self, cellDataObj):
        fig1 = plt.figure(figsize=(self.l, self.h))
        fig1_f1 = fig1.add_subplot(121)
        fig1_f1.plot(cellDataObj.time, cellDataObj.volt, "b", label="Voltage")
        fig1_f1.set_xlabel("Time [s]")
        fig1_f1.set_ylabel("Voltage [V]")
        fig1_f1.set_title("Voltage from \n" + cellDataObj.filename)
        fig1_f1.legend(loc="lower right")
        fig1_f1.grid(True)

        fig1_f2 = fig1.add_subplot(122)
        fig1_f2.plot(cellDataObj.time, cellDataObj.curr, "b", label="Current")
        fig1_f2.set_xlabel("Time [s]")
        fig1_f2.set_ylabel("Current [A]")
        fig1_f2.set_title("Current from \n" + cellDataObj.filename)
        fig1_f2.legend(loc="lower right")
        fig1_f2.grid(True)

    def plotComputedOCV(self, cellExtractOCVObj):
        fig2 = plt.figure(figsize=(self.l, self.h))

        fig2_f1 = fig2.add_subplot(111)
        fig2_f1.plot(cellExtractOCVObj.disTime, cellExtractOCVObj.disOCV, "r--", label="Discharge OCV")
        fig2_f1.plot(cellExtractOCVObj.chgTime, cellExtractOCVObj.chgOCV, "g--", label="Charge OCV")
        fig2_f1.plot(cellExtractOCVObj.disTime, cellExtractOCVObj.OCV, "b", label="Average OCV")
        fig2_f1.set_xlabel("Time [s]")
        fig2_f1.set_ylabel("Voltage [V]")
        fig2_f1.set_title("Average OCV from \n" + cellExtractOCVObj.filename)
        fig2_f1.legend()
        fig2_f1.grid(True)
    
    def plotLoadedOCV(self, cellSimObj):
        fig3 = plt.figure(figsize=(self.l, self.h))
        fig3_f1 = fig3.add_subplot(111)
        fig3_f1.plot(cellSimObj.SOCOCV, cellSimObj.voltOCV, "b", label="Average OCV")
        fig3_f1.set_xlabel("SOC [%]")
        fig3_f1.set_ylabel("Voltage [V]")
        fig3_f1.set_title("Loaded OCV from \n" + cellSimObj.filenameOCV)
        fig3_f1.legend()
        fig3_f1.grid(True)

    def plotDynamic(self, cellSimObj):
        fig4 = plt.figure(figsize=(self.l, self.h))
        # fig4_f1 = fig4.add_subplot(131)
        # fig4_f1.plot(cellSimObj.time, cellSimObj.testOCV, "b", label="Test OCV")
        # fig4_f1.set_xlabel("Time [sec]")
        # fig4_f1.set_ylabel("Voltage [V]")
        # fig4_f1.set_title("Test OCV from \n" + cellSimObj.filename)
        # fig4_f1.legend()
        # fig4_f1.grid(True)

        # fig4_f2 = fig4.add_subplot(121)
        # fig4_f2.plot(cellSimObj.time, cellSimObj.overPotVolt, "b", label="Overpotential")
        # fig4_f2.set_xlabel("Time [sec]")
        # fig4_f2.set_ylabel("Voltage [V]")
        # fig4_f2.set_title("Overpotential from \n" + cellSimObj.filename)
        # fig4_f2.legend()
        # fig4_f2.grid(True)

        fig4_f3 = fig4.add_subplot(111)
        fig4_f3.plot(cellSimObj.time, cellSimObj.vT, "b", label="Simulated Voltage")
        fig4_f3.plot(cellSimObj.time, cellSimObj.volt, "g--", label="True Voltage")
        fig4_f3.set_xlabel("Time [sec]")
        fig4_f3.set_ylabel("Voltage [V]")
        fig4_f3.set_title("Simulated Voltage and \n Terminal Voltage from " + cellSimObj.filename)
        fig4_f3.legend()
        fig4_f3.grid(True)