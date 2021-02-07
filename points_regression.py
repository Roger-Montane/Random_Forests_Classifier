#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:08:58 2018

@author: joans
"""

import numpy as np
import matplotlib.pyplot as plt

n_points = 200
rng = np.random.RandomState(1)
X = np.linspace(0, 6, n_points)[:, np.newaxis]
noise = rng.normal(0, 0.1, n_points)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + noise
idx = rng.uniform(size=n_points)>0.5
# half points for train, half for test
X_train, y_train = X[idx], y[idx]
X_test, y_test = X[np.logical_not(idx)], y[np.logical_not(idx)]

plt.figure()
plt.plot(X_train, y_train, '.:', label='train')
plt.plot(X_test, y_test, '.', label='test')
plt.legend()
