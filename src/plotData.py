import matplotlib
import matplotlib.pyplot as plt

# Define DPI of figures plotted
matplotlib.rcParams['figure.dpi'] = 100

class plotData():
    """

    Plots data from dataset, computed OCV, loaded OCV and simulated terminal voltage

    Args:
        None

    """
    def __init__(self):
        # Setup dimensions of the figures
        self.l = 10.0
        self.h = self.l*3/4

    def plotDataFromDataset(self, cellDataObj):
        """
        
        Figure 1: Plot true terminal voltage from cell dataset

        Args:
            self (plotData): Pointer to plotData class object
            cellDataObj (cellData): cellData class object
        Returns:
            None

        """
        fig1 = plt.figure(figsize=(self.l, self.h))

        fig1_f1 = fig1.add_subplot(111)
        fig1_f1.plot(cellDataObj.time, cellDataObj.volt, "b", label="Voltage")
        fig1_f1.set_xlabel("Time [s]")
        fig1_f1.set_ylabel("Voltage [V]")
        fig1_f1.set_title("Voltage from \n" + cellDataObj.filename)
        fig1_f1.legend(loc="lower right")
        fig1_f1.grid(True)

    def plotComputedOCV(self, cellExtractOCVObj):
        """
        
        Figure 2: Plot extracted cell OCVs

        Args:
            self (plotData): Pointer to plotData class object
            cellExtractOCVObj (cellExtractOCV): cellExtractOCV class object
        Returns:
            None

        """
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
    
    def plotLoadedOCV(self, cellTrainValidateObj):
        """
        
        Figure 3: Plot loaded OCV from saved file

        Args:
            self (plotData): Pointer to plotData class object
            cellTrainValidateObj (cellTrainValidate): cellTrainValidate class object
        Returns:
            None

        """
        fig3 = plt.figure(figsize=(self.l, self.h))

        fig3_f1 = fig3.add_subplot(111)
        fig3_f1.plot(cellTrainValidateObj.SOCOCV, cellTrainValidateObj.voltOCV, "b", label="Average OCV")
        fig3_f1.set_xlabel("SOC [%]")
        fig3_f1.set_ylabel("Voltage [V]")
        fig3_f1.set_title("Loaded OCV from \n" + cellTrainValidateObj.filenameOCV)
        fig3_f1.legend()
        fig3_f1.grid(True)

    def plotDynamic(self, cellTrainValidateObj):
        """
        
        Figure 4: Plot simulated terminal voltage and true terminal voltage

        Args:
            self (plotData): Pointer to plotData class object
            cellTrainValidateObj (cellTrainValidate): cellTrainValidate class object
        Returns:
            None

        """
        fig4 = plt.figure(figsize=(self.l, self.h))

        fig4_f1 = fig4.add_subplot(111)
        fig4_f1.plot(cellTrainValidateObj.time, cellTrainValidateObj.vT, "b", label="Simulated Voltage")
        fig4_f1.plot(cellTrainValidateObj.time, cellTrainValidateObj.volt, "g--", label="True Voltage")
        fig4_f1.set_xlabel("Time [s]")
        fig4_f1.set_ylabel("Voltage [V]")
        fig4_f1.set_title("Simulated Voltage using " + cellTrainValidateObj.filenameCellParamsOpti + " \n and \n True Voltage from " + cellTrainValidateObj.filename)
        fig4_f1.legend()
        fig4_f1.grid(True)

    def plotCostVisulaize(self, cellTrainValidateObj):
        """
        
        Figure 5: Plot visualized CRMSE cost function

        Args:
            self (plotData): Pointer to plotData class object
            cellTrainValidateObj (cellTrainValidate): cellTrainValidate class object
        Returns:
            None

        """
        fig5 = plt.figure(figsize=(self.l, self.h))

        fig5_f1 = fig5.add_subplot(111)
        fig5_f1.plot(cellTrainValidateObj.rc1, cellTrainValidateObj.visRMS, "+b", label="CRMSE for different RC1")
        # fig5_f1.plot(cellTrainValidateObj.visRMS, "+b", label="CRMSE for different RC1")
        fig5_f1.set_xlabel("RC1 time constant [s]")
        fig5_f1.set_ylabel("CRMSE [V]")
        fig5_f1.legend()
        fig5_f1.grid(True)    