# -*- coding=utf-8 -*-
#!/usr/bin/python
from numpy import *

class KalmanFilter(object):
    # ref. 《Probabilistic Robotics(cn)》 P31
    def __init__(self, data1, data2, data3 = 0):
        '''
        a simple kalman filter, before predict or correct, the matrices above should be initialized first:
        measurement_matrix---k*n matrix, k is the dimension of measurement vector z
        control_matrix---n*m matrix, m is the dimension of control vector u
        transition_matrix---n*n matrix, n is the dimension of state vector x
        process_noise_cov---n*n matrix
        measurement_noise_cov---k*k matrix
        :param data1: dimension of transition matrix
        :param data2: dimension of measurement matrix
        :param data3: dimension of control matrix
        '''
        self.__transition_dim = data1

        self.__last_x = mat(zeros((int(data1),1)))
        self.__last_var = mat(identity(int(data1)))
        self.__xhat = mat(zeros((int(data1),1)))
        self.__vhat = mat(identity(int(data1)))
        self.__I = mat(identity(int(data1)))

        self.__measurement_dim = data2

        self.__control_dim = data3

    def setMeasurementMatrix(self, matrix):
        self.__measurement_matrix = matrix

    def setControlMatrix(self, matrix):
        self.__control_matrix = matrix

    def setTransitionMatrix(self, matrix):
        self.__transition_matrix = matrix

    def setProcessNoiseCov(self, matrix):
        self.__process_noisy_matrix = matrix

    def setMeasurementNoisyCov(self, matrix):
        self.__measurement_noisy_cov = matrix

    def predict(self, u):
        '''
        Predict
        :param u: control matrix
        :return: predict state

            miu_hat = A * miu_minus + B * u
            var_hat = A * var_minus + A^T + R
        '''
        self.__xhat = self.__transition_matrix * self.__last_x + self.__control_matrix * u
        self.__vhat = self.__transition_matrix * self.__last_var * self.__transition_matrix.T + self.__process_noisy_matrix
        return self.__xhat

    def correct(self, measurement):
        '''

        :param measurement: measurement matrix
        :return:

            K = var_hat * C^T * (C * var_hat * C^T + Q)^-1
            miu = miu_hat + K * (z - C * miu_hat)
            var = (I - K * C) * var_hat
        '''

        k_t = self.__vhat * self.__measurement_matrix.T * (self.__measurement_matrix * self.__vhat
                                                           * self.__measurement_matrix.T + self.__measurement_noisy_cov).I

        self.__last_x = self.__xhat + k_t * (measurement - self.__measurement_matrix * self.__xhat)
        self.__last_var = (self.__I - k_t * self.__measurement_matrix) * self.__vhat

        return self.__last_x
