#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:28:15 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import numpy
import pandas
import os
import time
import sys
from GaussianEmulator import GaussianEmulator

sys.path.append(os.environ["LESMAINSCRIPTS"])
from FileSystem import FileSystem

class CloudEmulatorInterface(GaussianEmulator):
    def __init__(self, configFileYAML):
        self.configFile = FileSystem.readYAML( configFileYAML )

        self.__init__readTrainingData()
        self.__init__readPredictionInput()

        super().__init__( self.trainingData )

    def __init__readTrainingData(self):
        trainingData = self.__readCSVfile( self.configFile["trainingDataInputFile"] )

        self.trainingData = trainingData.values

    def __init__readPredictionInput(self):
        predictionInput = self.__readCSVfile( self.configFile["predictionDataInputFile"] )

        self.predictionInput = predictionInput.values[:,:-2]

    def __readCSVfile(self, file):
        return pandas.read_csv(file, delim_whitespace = True, header = None)

    def predictEmulator(self):
        super().predictEmulator( self.predictionInput )

    def savePredictions(self):
        numpy.savetxt( self.configFile["predictionOutputFile"], self.predictions )

def main(file):

    assert( os.path.isfile(file))

    emulator = CloudEmulatorInterface( file )

    emulator.main()

    emulator.predictEmulator()

    emulator.savePredictions()


if __name__ == "__main__":
    start = time.time()

    if len(sys.argv) >= 2:
        file = sys.argv[1]
    else:
        file = "config.yaml"

    main( file )

    end = time.time()
    print(f"\nEmulator completed in { end - start : .1f} seconds")
