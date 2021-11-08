import matplotlib.pyplot as plt
import numpy
import random

import xgboost

nSample = 1001

# Generate the composite sine curve
random.seed(a = 20181128)

x = numpy.zeros((nSample,1))
y = numpy.zeros((nSample,1))

for i in range(nSample):
    x[i] = (i - 500.0) / 500.0
    y[i] = 1.0 - 3.0 * numpy.sin(2.0 * x[i] + 1.0) + 5.0 * numpy.sin(20.0 * x[i] - 2.0);

# Plot the data
plt.figure(figsize=(10,6))
plt.scatter(x, y, c = 'green', s = 2)
plt.grid(axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Perform the Gradient Boosting
_objXGB = xgboost.XGBRegressor(max_depth = 3, n_estimators = 1000, verbosity = 1,
                               objective = 'reg:squarederror', booster = 'gbtree',
                               random_state = 60616)

fit_gbm = _objXGB.fit(x, y.ravel())

predY_gbm = fit_gbm.predict(x)
Rsq_gbm = fit_gbm.score(x, y.ravel())
print('XGBoosting Model:', 'R-Squared = ', Rsq_gbm)

# Plot the data
plt.figure(figsize=(10,6))
plt.scatter(x, predY_gbm, c = 'blue', s = 2, label = 'XGB 1000')
plt.scatter(x, y, c = 'green', s = 2, label = 'Data')
plt.grid(axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
