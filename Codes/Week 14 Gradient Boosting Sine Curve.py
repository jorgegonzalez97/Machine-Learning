import matplotlib.pyplot as plt
import numpy
import sklearn.ensemble as ensemble
import sklearn.linear_model as lm

nSample = 1001

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

# Build a linear regression
regC = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
fitC = regC.fit(x, y)
predYC = regC.predict(x)
RSqC = regC.score(x, y)
print('Complete Model :', 'R-Squared = ', RSqC, '\n Intercept = ', regC.intercept_, '\n Coefficients = ', regC.coef_)

# Perform the Gradient Boosting
gbm = ensemble.GradientBoostingRegressor(loss='ls', criterion='mse', n_estimators = 500,
                                         max_leaf_nodes = 14, verbose=1)
fit_gbm = gbm.fit(x, y.ravel())
predY_gbm = gbm.predict(x)
Rsq_gbm = gbm.score(x, y.ravel())
print('Gradient Boosting Model:', 'R-Squared = ', Rsq_gbm)

# Plot the data
plt.figure(figsize=(10,6))
plt.scatter(x, predY_gbm, c = 'blue', s = 2, label = 'GBM 500')
plt.scatter(x, y, c = 'green', s = 2, label = 'Data')
plt.grid(axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()