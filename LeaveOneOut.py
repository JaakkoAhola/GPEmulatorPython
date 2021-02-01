#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:29:14 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import pandas
import numpy
import os
import sys

sys.path.append(os.environ["PYTHONEMULATOR"])
from GaussianEmulator import GaussianEmulator

class LeaveOneOut:

    def loopLeaveOneOut(matrix, indexName, optimization = {"maxiter": 30000, "n_restarts_optimizer" : 10}, boundOrdo = [1e-3, 1e12] ):
        train = matrix.drop(indexName).values
        emulator = GaussianEmulator(train,
                                    maxiter = optimization["maxiter"],
                                    n_restarts_optimizer = optimization["n_restarts_optimizer"],
                                    boundOrdo = boundOrdo)
        emulator.main()
        predict = matrix.loc[indexName].values[:-2].reshape(1,-1)
        emulator.predictEmulator( predict )
        emulatedValue = emulator.getPredictions().item()
        
        return emulatedValue