# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:07:21 2023

@author: NOA
"""

import numpy as np

class AHRS:
    def __init__(self, SamplePeriod, Kp, Ki, KpInit):
        self.SamplePeriod = SamplePeriod
        self.Kp = Kp
        self.Ki = Ki
        self.KpInit = KpInit
        self.q = np.array([-0.5, 0.5, 0.5, -0.5])  # Initial quaternion
        self.IntError = np.zeros(3)
        
    def quaternProd(self, a, b):
        ab = np.zeros(4)
        ab[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        ab[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        ab[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        ab[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return ab

    def quaternConj(self, q):
        qConj = np.zeros(4)
        qConj[0] = q[0]
        qConj[1:] = -q[1:]
        return qConj

    def UpdateIMU(self, Gyroscope, Accelerometer):
        # Normalise accelerometer measurement
        if np.linalg.norm(Accelerometer) == 0:
            print('Accelerometer magnitude is zero. Algorithm update aborted.')
            return
        else:
            Accelerometer = Accelerometer / np.linalg.norm(Accelerometer)

        # Compute error between estimated and measured direction of gravity
        v = np.array([2 * (self.q[1] * self.q[3] - self.q[0] * self.q[2]),
                      2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]),
                      self.q[0]**2 - self.q[1]**2 - self.q[2]**2 + self.q[3]**2])  # estimated direction of gravity
        error = np.cross(v, Accelerometer)

        self.IntError = self.IntError + error  # compute integral feedback terms (only outside of init period)

        # Apply feedback terms
        Ref = Gyroscope - (self.Kp * error + self.Ki * self.IntError)

        # Compute rate of change of quaternion
        #pDot = 0.5 * self.quaternProd(self.q, [0, Ref[0], Ref[1], Ref[2]])  # compute rate of change of quaternion
        pDot = 0.5 * self.quaternProd(self.q, np.array([0, Ref[0], Ref[1], Ref[2]]))

        self.q = self.q + pDot * self.SamplePeriod  # integrate rate of change of quaternion
        
        self.q = self.q / np.linalg.norm(self.q)  # normalise quaternion

        # Store conjugate
        self.Quaternion = self.quaternConj(self.q)
        return self.q
