#############################################
##  Original Author: Tomás Pelayo Benedet  ##
##  Email:   tomaspelayobenedet@gmail.com  ##
##  Last Modified:           May 30, 2023  ##
#############################################

from DGAFramework.DatasetManager.DatasetManager import DatasetManager
from DGAFramework.DataElement.DataElement import DataElement
from test.classes.frameworkTest.dataElementFrameworkTest import DataElementFrameworkTest

################################################################################
#  CODE  #######################################################################
################################################################################

class DatasetManagerFrameworkTest(DatasetManager):

    def parseDataElement(self, line:str) -> DataElement:
        domain = line.split(";")[0]
        isDGAstr = line.split(";")[1]

        isDGA = False
        if (eval(isDGAstr)):
            isDGA = True

        return DataElementFrameworkTest(domain, isDGA)