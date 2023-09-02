#############################################
##  Original Author: Tomás Pelayo Benedet  ##
##  Email:   tomaspelayobenedet@gmail.com  ##
##  Last Modified:           May 29, 2023  ##
#############################################

from DGAFramework.Result.Result import Result

class ResultFrameworkTest(Result):

    accuracy = None

    def __init__(self, accuracy) -> None:
        self.accuracy = accuracy

    def print(self):
        print("Accuracy -> " + str(self.accuracy))
        pass
    
    def toCSVheader(self, separator:str) -> str:
        '''toCSVheader function'''
        pass

    def toCSV(self, separator:str) -> str:
        '''toCSV function'''
        pass