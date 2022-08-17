#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:41:10 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""

import numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class GaussianEmulator:

    def __init__(self, maxiter=10000, n_restarts_optimizer=10, boundOrdo=[1e-12, 1e12]):
        # Standardize inputs and outputs
        # NOTE! GP must have standardized values always!
        # GP output standardization is hardcoded, input not but it would be strange to not have it on...

        self.maxiter = maxiter

        self.n_restarts_optimizer = n_restarts_optimizer

        self.boundOrdo = tuple(boundOrdo)

    def optimizer(self, obj_func, initial_theta, bounds=None):
        opt_res = minimize(obj_func,
                           initial_theta,
                           method="L-BFGS-B",
                           options={"maxiter": self.maxiter},
                           bounds=bounds,
                           jac=True)

        theta_opt, func_min = opt_res.x, opt_res.fun

        return theta_opt, func_min

    def fit(self, inputs, target):
        # Original LES data used for training
        self.inputMatrix = inputs
        # Change rainrate units to match satellite rainrate
        self.targetMatrix = target
        self.numberOfInputVariables = self.inputMatrix.shape[1]

        self.theta = numpy.array([0.9650, 0.6729, 3.5576, 4.7418, 1.2722, 4.0612, 0.5, 2.4, 4.3, 3.2, 1.5, 0.5, 2.4, 4.3, 3.2])[: self.numberOfInputVariables]

        self.getScaled()
        self.setDefaultKernel()
        self.trainEmulator()

        return self.getEmulator()

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
        self.targetMatrix = self.scaler_out.transform(self.targetMatrix.reshape(-1, 1))

        self.scaler_out_gp = self.scaler_out

    def setDefaultKernel(self):

        self.__checkTheta()
        self.kernel = C(0.9010, self.boundOrdo) \
            + RBF(self.theta, self.boundOrdo) \
            + WhiteKernel(noise_level=0.346, noise_level_bounds=self.boundOrdo)

    def setKernel(self, kernel):
        self.kernel = kernel

    def trainEmulator(self):
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=self.n_restarts_optimizer,
                                           alpha=1e-10,
                                           normalize_y=False,
                                           optimizer=self.optimizer,
                                           random_state=0)
        self.gp.fit(self.inputMatrix, self.targetMatrix)

    def getEmulator(self):
        return self.gp

    def getScaledPredictionMatrix(self, predictionMatrix):
        predictionMatrixScaled = self.scaler.transform(predictionMatrix)

        return predictionMatrixScaled

    def getScaledTargetMatrix(self, targetMatrix):
        targetScaled = self.scaler_out.transform(targetMatrix)

        return targetScaled

    def getNonScaledPredictions(self, predictions):
        return self.scaler_out_gp.inverse_transform(predictions)

    def predict(self, predictionMatrix):

        predictionMatrixScaled = self.scaler.transform(predictionMatrix)

        predictions = self.gp.predict(predictionMatrixScaled, return_std=False)
        self.predictions = self.scaler_out_gp.inverse_transform(predictions)

        return self.predictions
