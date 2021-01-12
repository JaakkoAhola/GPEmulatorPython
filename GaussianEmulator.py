#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:41:10 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GaussianEmulator:
    
    def __init__(self, trainingData):
        # Standardize inputs and outputs
        # NOTE! GP must have standardized values always!
        # GP output standardization is hardcoded, input not but it would be strange to not have it on...
        
        # Original LES data used for training
        self.inputMatrix = trainingData[:, :-2]
        # Change rainrate units to match satellite rainrate
        self.targetMatrix = trainingData[:, -1]
        
        self.numberOfInputVariables = self.inputMatrix.shape[1]
        
        
        self.theta = numpy.array([0.9650, 0.6729, 3.5576, 4.7418, 1.2722, 4.0612, 0.5, 2.4, 4.3, 3.2, 1.5, 0.5, 2.4, 4.3, 3.2])[: self.numberOfInputVariables ]
        
    def main(self):
        self.getScaled()
        self.setDefaultKernel()
        self.trainEmulator()
        
    def setTheta(self, theta):
        self.theta = theta
    
    def __checkTheta(self):
        assert(len(self.theta) == self.numberOfInputVariables)
        
    def getLogaritmOfTargets(self):
        """
        Optional method to be used

        """
        self.targetMatrix = numpy.log(self.targetMatrix)

    def getScaled(self):
        self.scaler = StandardScaler().fit(self.inputMatrix)
        self.inputMatrix = self.scaler.transform(self.inputMatrix)
    
        self.scaler_out = StandardScaler().fit(self.targetMatrix.reshape(-1, 1))
        self.targetMatrix = self.scaler_out.transform(self.targetMatrix[:, None])

        self.scaler_out_gp = self.scaler_out
        
    def setDefaultKernel(self):
        
        self.__checkTheta()
        self.kernel = C(0.9010, (1e-12, 1e12)) \
                        + RBF(self.theta, (1e-12, 1e12)) \
                        + WhiteKernel(noise_level=0.346, noise_level_bounds=(1e-12, 1e+12))
                        
    def setKernel(self, kernel):
        self.kernel = kernel
    
    def trainEmulator(self):
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=0, normalize_y=False)
        self.gp.fit(self.inputMatrix, self.targetMatrix)
    
    def predictEmulator(self, predictionMatrix):
        
        predictions = self.gp.predict(predictionMatrix, return_std=False)
        self.predictions = self.scaler_out_gp.inverse_transform( predictions )
        
    def getPredictions(self):
        return self.predictions
    
