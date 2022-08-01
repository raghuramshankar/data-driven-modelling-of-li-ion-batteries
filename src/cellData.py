import numpy as np
import pandas as pd


class cellData:
    """
    
    Extracts dataframe from dataset csv found in the pathname and filename input
    Creates a column for time of experiment based on progTime
    Assigns class variables for voltage, current and discharge capacity, sample time
    and coulombic efficiency found from dataset

    Args:
        None

    """
    def __init__(self, filename, pathname):
        # Copy filename and pathname to class variables
        self.pathname = pathname
        self.filename = filename

        # Create fullname class variable by concatenating pathname and filename
        self.fullname = self.pathname + self.filename

    def convertToSec(self, progTime):
        """
        
        Converts time in hours to seconds

        Args:
            self (cellData): Pointer to cellData class object
            progTime (string): experiment progress time in hours
        Returns:
            expression (float): time in seconds
        
        """
        [h, m, s] = map(float, progTime.split(":"))
        return h * 3600 + m * 60 + s

    def extractData(self):
        """
        
        Extracts dataframe from dataset csv and assigns class variables

        Args:
            self (cellData): Pointer to cellData class object
        Returns:
            None
        
        """
        # Skip first 28 rows
        self.df = pd.read_csv(self.fullname, skiprows=28, dtype=str)

        # Clean up self.df
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]
        self.df = self.df.drop(0)

        # Convert self.df to numpy df
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        # Compute time in seconds and add to self.df
        self.progTime = [
            self.convertToSec(progTime) for progTime in self.df["Prog Time"]
        ]
        self.time = [progTime - self.progTime[0] for progTime in self.progTime]
        self.df["Time"] = [time for time in self.time]

        # Create variables
        self.volt = np.asarray([voltage for voltage in self.df["Voltage"]])
        self.curr = np.asarray([-current for current in self.df["Current"]])
        self.disCap = np.asarray([capacity for capacity in self.df["Capacity"]])
        self.dt = np.mean(np.diff(self.time))
        self.eta = 1.0

        print("extract data done from", self.filename)
