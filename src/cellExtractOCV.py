import numpy as np
import pandas as pd


class cellExtractOCV:
    """
    
    Extracts OCV data from dataframe and computes OCV

    Args:
        None
    
    """
    def __init__(self, cellDataObj):
        """
        
        Initializes cellExtractOCV class object by copying data from cellDataObj

        Args:
            self (cellExtractOCV): Pointer to cellExtractOCV class object
            cellDataObj (cellData): cellData class object
        
        """
        # Copy the cellDataObj dataframe to self
        self.df = cellDataObj.df

        # Copy the cellDataObj filename to self
        self.filename = cellDataObj.filename

    def extractOCV(self):
        """
        
        Extract OCV data from dataframe
        Uses Status column to determine different stages of the OCV experiment

        Args:
            self (cellExtractOCV): Pointer to cellExtractOCV object
        Returns:
            None
        
        """
        self.disOCV = [
            self.df["Voltage"].to_numpy()[i]
            for i in range(len(self.df))
            if self.df["Status"].to_numpy()[i] == "DCH"
        ]
        self.disTime = [
            self.df["Time"].to_numpy()[i]
            for i in range(len(self.df))
            if self.df["Status"].to_numpy()[i] == "DCH"
        ]
        self.chgOCV = np.flip(
            [
                self.df["Voltage"].to_numpy()[i]
                for i in range(len(self.df))
                if self.df["Status"].to_numpy()[i] == "CHA"
            ]
        )
        self.chgTime = [
            self.df["Time"].to_numpy()[i]
            for i in range(len(self.df))
            if self.df["Status"].to_numpy()[i] == "CHA"
        ]
        self.chgTime = self.chgTime - self.chgTime[0]
        self.pauOCV = [
            self.df["Voltage"].to_numpy()[i]
            for i in range(len(self.df))
            if self.df["Status"].to_numpy()[i] == "PAU"
        ]
        self.disCap = [
            self.df["Capacity"].to_numpy()[i]
            for i in range(len(self.df))
            if self.df["Status"].to_numpy()[i] == "DCH"
        ]

        print("extract OCV done")

    def computeOCV(self):
        """
        
        Compute OCV by clipping the Charge OCV data to the length of discharge OCV data
        Compute SOC using discharge capacity and charge discharged

        Args:
            self (cellExtractOCV): Pointer to cellExtractOCV class object
        Returns:
            None
        
        """
        self.OCV = (self.disOCV + self.chgOCV[0 : len(self.disOCV)]) / 2
        self.disCapacity = -self.disCap[-1]
        self.SOC = np.flip(np.negative(self.disCap) / self.disCapacity)

        print("compute OCV done")

    def saveOCV(self):
        """
        
        Saves the OCV-SOC data in a different dataframe

        Args:
            self (cellExtractOCV): Pointer to cellExtractOCV class object
        Returns:
            None
        
        """
        self.dfOCV = {}
        self.dfOCV.update({"time": self.chgTime[0 : len(self.disOCV)]})
        self.dfOCV.update({"OCV": self.OCV})
        self.dfOCV.update({"SOC": self.SOC})
        self.dfOCV.update(
            {"disCapacity": [self.disCapacity for _ in range(len(self.OCV))]}
        )
        self.dfOCV = pd.DataFrame(self.dfOCV)
        self.dfOCV.to_csv(
            "results/OCV--" + self.filename.replace("/", "--"), index=False
        )

        print("save OCV done")

    def runOCV(self):
        """
        
        Runs all the OCV-SOC functions in this class

        Args:
            self (cellExtractOCV): Pointer to cellExtractOCV class object
        Returns:
            None
        
        """
        self.extractOCV()
        self.computeOCV()
        self.saveOCV()
