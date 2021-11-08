import matplotlib.pyplot as plt
import numpy

import graphviz
import sklearn.metrics as metrics
import sklearn.tree as tree
import statsmodels.api as stats

nSample = 1001

# Generate (x,y) where logit(y) = a + b*x
numpy.random.seed(seed = 20191120)

x = numpy.zeros((nSample,1))
y = numpy.zeros(nSample)

for i in range(nSample):
    x[i] = (i - 200.0) / 100.0
    u = 5 - 0.1 * x[i]
    p = numpy.exp(u)
    p = 1 / (1.0 + p)
    y[i] = numpy.random.binomial(1, p)
   
# Descriptives
print('Mean of x = ', x.mean(), 'STD of x = ', x.std())
print('Y = 0, Mean of x = ', x[y==0].mean(), 'STD of x = ', x[y==0].std())
print('Y = 1, Mean of x = ', x[y==1].mean(), 'STD of x = ', x[y==1].std())

unq_y, cnt_y = numpy.unique(y, return_counts = True)
print(unq_y, cnt_y)

# Plot the data
plt.figure(figsize=(10,6))
plt.scatter(x = x, y = y)
plt.grid(axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

threshPredProb = numpy.mean(y)

# Build an Intercept-only regression model (i.e., y-hat = mean of y) on the training sample
# Model 0 is Origin = Intercept
X = x
X = stats.add_constant(X, prepend=True)
logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
yPredProb = thisFit.predict(X)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter))

yPredClass = numpy.where(yPredProb[:,1] >= threshPredProb, 1, 0)
y_accuracy = metrics.accuracy_score(y, yPredClass)
print("Accuracy Score = ", y_accuracy)

residP = y - yPredProb[:,1]

_objTree = tree.DecisionTreeRegressor(criterion = 'mse', max_depth = 2, random_state = 60616)
thisTree = _objTree.fit(x, residP)
residPred = thisTree.predict(x)

dot_data = tree.export_graphviz(thisTree, out_file = None,
                                impurity = True, filled = True,
                                feature_names = ['x'],
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)

graph

yPredProb_1 = numpy.minimum(1.0, numpy.maximum(0.0, y - residPred))

print(yPredProb_1.min())
print(yPredProb_1.max())

yPredClass_1 = numpy.where(yPredProb_1 >= threshPredProb, 1, 0)
y_accuracy = metrics.accuracy_score(y, yPredClass_1)
print("Accuracy Score = ", y_accuracy)


