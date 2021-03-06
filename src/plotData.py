import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 100

class plotData():
    def __init__(self):
        self.l = 10.0
        self.h = self.l*3/4

    def plotDataFromDataset(self, cellDataObj):
        fig1 = plt.figure(figsize=(self.l, self.h))
        fig1_f1 = fig1.add_subplot(111)
        fig1_f1.plot(cellDataObj.time, cellDataObj.volt, "b", label="Voltage")
        fig1_f1.set_xlabel("Time [s]")
        fig1_f1.set_ylabel("Voltage [V]")
        fig1_f1.set_title("Voltage from \n" + cellDataObj.filename)
        fig1_f1.legend(loc="lower right")
        fig1_f1.grid(True)

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

        fig4_f1 = fig4.add_subplot(111)
        fig4_f1.plot(cellSimObj.time, cellSimObj.vT, "b", label="Simulated Voltage")
        fig4_f1.plot(cellSimObj.time, cellSimObj.volt, "g--", label="True Voltage")
        fig4_f1.set_xlabel("Time [s]")
        fig4_f1.set_ylabel("Voltage [V]")
        fig4_f1.set_title("Simulated Voltage using " + cellSimObj.filenameCellParamsOpti + " \n and \n True Voltage from " + cellSimObj.filename)
        fig4_f1.legend()
        fig4_f1.grid(True)