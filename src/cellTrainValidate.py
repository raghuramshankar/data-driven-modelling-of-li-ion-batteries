import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class cellTrainValidate:
    """
    
    Computes the RC cell model parameters using loaded OCV data and training dynamic data
    Validates the RC cell model parameters by simulating using the validation dynamic data
    Computes the CRMSE between the simulated cell and true validation dynamic data

    Args:
        None
    
    """
    def __init__(self, cellDataObj):
        """
        
        Initializes the cellSim class by copying data from cellDataObj
        Clips the true cell voltage to the time of experiment

        Args:
            self (cellSim): Pointer to cellSim class object
            cellDataObj (cellData): cellData class object

        """
        self.filename = cellDataObj.filename
        self.time = cellDataObj.time
        self.volt = cellDataObj.volt
        self.curr = cellDataObj.curr
        self.dt = cellDataObj.dt
        self.eta = cellDataObj.eta
        self.nRC = 2
        self.nTime = len(cellDataObj.time)
        self.volt = self.volt[0: self.nTime]
        self.sign = np.zeros_like(self.curr)

    def loadOCV(self):
        """
        
        Loads the extracted OCV-SOC data from preselected index and saves it as class variables

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        pathname = "results/"
        filenames = [
            filename for filename in os.listdir(pathname) if filename.startswith("OCV")
        ]


        index = 0
        self.filenameOCV = filenames[index]
        self.dfOCV = pd.read_csv(pathname + self.filenameOCV)
        self.timeOCV = self.dfOCV["time"].to_numpy()
        self.voltOCV = self.dfOCV["OCV"].to_numpy()
        self.SOCOCV = self.dfOCV["SOC"].to_numpy()
        self.capacityOCV = self.dfOCV["disCapacity"].to_numpy()[0]

        print("load OCV done")

    def extractDynamic(self):
        """
        
        Computes the overpotential between true terminal voltage and estimated OCV at each datapoint

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        # Find the initial SOC value by:
        # 1. Find the initial terminal voltage (assumed as OCV)
        # 2. Find the index of the initial terminal voltage in the OCV-SOC table
        # 3. Find the value of SOC using the above index value
        self.initSOC = self.SOCOCV[np.argmin(abs(self.voltOCV - self.volt[0]))]

        # Compute the test SOC values by using the coulomb counting formula
        self.testSOC = self.initSOC - self.dt / (
            self.capacityOCV * 3600
        ) * self.eta * np.cumsum(self.curr)

        # Compute the test OCV values by using a similar method to finding the initial SOC
        self.testOCV = [
            self.voltOCV[np.argmin(abs(self.SOCOCV - soc))] for soc in self.testSOC
        ]

        # Compute the overpotential voltage by finding the difference between true terminal voltage and test OCV voltage 
        self.overPotVolt = self.volt - self.testOCV
        print("extract dynamic done")

    def loadCellParamsOpti(self):
        """
        
        Loads the saved trained cell parameters and saves them as class variables

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        pathname = "results/"
        filenames = [
            filename for filename in os.listdir(pathname) if filename.startswith("CellParams")
        ]

        # Select the cell parameters to load
        index = 1
        self.filenameCellParamsOpti = filenames[index]
        self.dfCellParamsOpti = pd.read_csv(
            pathname + self.filenameCellParamsOpti)
        self.r0 = self.dfCellParamsOpti["r0"].to_numpy()
        self.r1 = self.dfCellParamsOpti["r1"].to_numpy()
        self.r2 = self.dfCellParamsOpti["r2"].to_numpy()
        self.c1 = self.dfCellParamsOpti["c1"].to_numpy()
        self.c2 = self.dfCellParamsOpti["c2"].to_numpy()

        print("load CellParamsOpti done from " + self.filenameCellParamsOpti)

    def computeRMS(self):
        """
        
        Computes the CRMSE between training/validation terminal voltage and true terminal voltage

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        self.rmsError = 1000 * np.sqrt(np.mean(np.square(self.vT - self.volt)))

        return self.rmsError

    def cellSim(self):
        """
        
        Simulates the terminal voltage for the entire experiment time using the estimated cell parameters

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        # Allocate memory for simulation variables
        self.iR = np.zeros((self.nRC, self.nTime))
        self.vC = np.zeros((self.nRC, self.nTime))
        self.vT = np.zeros(self.nTime)

        # Initialize the first element of terminal voltage array
        self.vT[0] = self.testOCV[0]

        # Compute f matrix from dt and RC parameters
        self.f = [
            np.exp(-self.dt / np.dot(self.r1, self.c1)),
            np.exp(-self.dt / np.dot(self.r2, self.c2))
        ]

        # Compute a and b matrices from f matrix
        self.aRC = np.diag(self.f)
        self.bRC = np.ones(self.nRC) - self.f

        # Simulate the cell
        for k in range(self.nTime - 1):
            self.iR[:, k + 1] = (
                np.dot(self.aRC, self.iR[:, k]) + self.bRC * self.curr[k]
            )
            self.vC[0, k] = self.iR[0, k] * self.r1
            self.vC[1, k] = self.iR[1, k] * self.r2
            self.vT[k + 1] = (
                self.testOCV[k] - np.sum(self.vC[:, k]) -
                self.curr[k] * self.r0
            )

    def printCellParams(self):
        """
        
        Prints the estimated cell parameters from training data
        
        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None

        """
        print("R0 = ", self.r0, "ohm")
        print("R1 = ", self.r1, "ohm")
        print("R2 = ", self.r2, "ohm")
        print("C1 = ", self.c1, "farad")
        print("C2 = ", self.c2, "farad")

    def objFn(self, x0):
        """
        
        Objective function for estimating cell params from training data
        Initializes RC params from optimization function
        Computes CRMSE and uses that as goodness of fit

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
            x0 (float array): Cell RC parameter array [R0, R1, R2, C1, C2]
        Returns:
            rmsError (float): CRMSE of simulated terminal voltage versus true terminal voltage 
        
        """
        self.r0 = x0[0]
        self.r1 = x0[1]
        self.r2 = x0[2]
        self.c1 = x0[3] * self.scaleFactorC
        self.c2 = x0[4] * self.scaleFactorC
        self.cellSim()
        rmsError = self.computeRMS()
        return rmsError

    def optFn(self):
        """
        
        Optimization function for parameter extraction
        Uses scale factors for resistances and capacitances to converge quicker and better
        Uses upper and lower bounds for resistances and capacitances in the optimization function
        Uses scipy.minimize to perform gradient descent using CRMSE to measure goodness of fit

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        print("started parameter extraction via optimization")

        # Define scale factors for RC parameters
        self.scaleFactorC = 1e3
        x0 = [1e-3, 1e-2, 1e-2, 100e3 /
              self.scaleFactorC, 100e3/self.scaleFactorC]

        # Define bounds for RC parameters
        bndsR0 = (0.1e-3, 50e-3)
        bndsR = (1e-3, 5000e-3)
        bndsC1 = (1e3/self.scaleFactorC, 500e3/self.scaleFactorC)
        bndsC2 = (1e3/self.scaleFactorC, 5000e3/self.scaleFactorC)
        bnds = (bndsR0, bndsR, bndsR, bndsC1, bndsC2)

        # Run the optimization function
        minimize(self.objFn, x0, method="SLSQP", bounds=bnds)

    def saveCellParamsOpti(self):
        """
        
        Saves the obtained optimized cell parameters as class variables, dataframe and csv

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        self.filenameCellParamsOpti = "results/CellParams--" + \
            self.filename.replace("/", "--")
        self.dfCellParams = {}
        self.dfCellParams.update({"r0": self.r0})
        self.dfCellParams.update({"r1": self.r1})
        self.dfCellParams.update({"r2": self.r2})
        self.dfCellParams.update({"c1": self.c1})
        self.dfCellParams.update({"c2": self.c2})
        self.dfCellParams = pd.DataFrame(self.dfCellParams, index=[0])
        self.dfCellParams.to_csv(self.filenameCellParamsOpti, index=False)

    def runSimTrain(self):
        """
        
        Calls class functions to train cell parameters by simulation and optimization
        Also finds CRMSE with true terminal voltage

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        print("starting training of RC2 cell model")
        self.loadOCV()
        self.extractDynamic()
        self.optFn()
        self.saveCellParamsOpti()
        self.printCellParams()
        self.cellSim()
        print("CRMSE = ", self.computeRMS(), "mV")

    def runSimValidate(self):
        """
        
        Calls class functions to validate cell parameters by simulating and 
        finding CRMSE with true terminal voltage

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        print("starting validation of RC2 cell model")
        self.loadOCV()
        self.extractDynamic()
        self.loadCellParamsOpti()
        self.printCellParams()
        self.cellSim()
        print("CRMSE = ", self.computeRMS(), "mV")
    
    def costVisualize(self):
        """
        
        Computes variables to plot and visualize CRMSE cost function

        Args:
            self (cellTrainValidate): Pointer to cellTrainValidate class object
        Returns:
            None
        
        """
        # Define range for input
        bndsR0 = 1e-1
        bndsR1 = 1e-1
        bndsR2 = 1e-1
        bndsC1 = 1e4
        bndsC2 = 1e4

        # Sample inputs uniformly
        visR0 = np.arange(0, bndsR0, 0.1)
        visR1 = np.arange(0, bndsR1, 0.1)
        visR2 = np.arange(0, bndsR2, 0.1)
        visC1 = np.arange(0, bndsC1, 0.1)
        visC2 = np.arange(0, bndsC2, 0.1)

        # Initialize CRMSE variable
        visRMS = np.empty([len(visR0, 1)])

        for j in range(len(visR0)):
            self.r0 = visR0[j]
            self.r1 = visR1[j]
            self.r2 = visR2[j]
            self.c1 = visC1[j]
            self.c2 = visC2[j]

            self.cellSim()
            visRMS[j] = self.computeRMS()
            