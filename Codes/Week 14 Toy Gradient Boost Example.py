import matplotlib.pyplot as plt
import numpy

import sklearn.metrics as metrics
import sklearn.tree as tree
import statsmodels.api as stats

x_train = numpy.array([[0.1, 0.3],
                       [0.2, 0.2],
                       [0.3, 0.1],
                       [0.4, 0.4],
                       [0.5, 0.7],
                       [0.6, 0.5],
                       [0.7, 0.9],
                       [0.8, 0.8],
                       [0.8, 0.2],
                       [0.9, 0.8]], dtype = float)

y_train = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype = float)

carray = ['blue', 'red']
plt.figure(figsize=(16,9))
i0 = numpy.where(y_train == 0)
plt.scatter(x_train[i0,0], x_train[i0,1], c = carray[0], label = 0, s = 100)
i1 = numpy.where(y_train == 1)
plt.scatter(x_train[i1,0], x_train[i1,1], c = carray[1], label = 1, s = 100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend(title = 'y', fontsize = 12, markerscale = 1, loc = 'upper left')
plt.show()

X = stats.add_constant(x_train, prepend=True)
logit = stats.MNLogit(y_train, X)
thisFit = logit.fit(method = 'bfgs', maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
yPredProb = thisFit.predict(X)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter))

threshPredProb = numpy.mean(y_train)
yPredClass = numpy.where(yPredProb[:,1] >= threshPredProb, 1, 0)
y_accuracy = metrics.accuracy_score(y_train, yPredClass)
print("Accuracy Score = ", y_accuracy)

# Iteration 0
currentP = yPredProb[:,1]
objFunc = numpy.sum(numpy.log(numpy.where(y_train == 1, currentP, 1.0-currentP)))
print('Objective Function = ', objFunc)
print('Predicted Event Probability:\n', yPredProb[:,1])

for iter in range(10):
   print('Iteration = ', iter+1)
   pseudoY = 2.0 * numpy.where(y_train == 1, 1.0-currentP, -currentP)

   # Set maximum number of leaves to 4
   regTree = tree.DecisionTreeRegressor(criterion='mse', max_leaf_nodes=4, random_state=60616)
   treeFit = regTree.fit(x_train, pseudoY)
   treePredProb = regTree.predict(x_train)

   # Find step
   qGoodStep = 0
   for step in numpy.arange(1.0,0.0,-0.1):
      nextP = currentP + step * treePredProb
      nextP = numpy.where(nextP >= 1.0, 1.0, nextP)
      nextP = numpy.where(nextP <= 0.0, 0.0, nextP)
      objFunc1 = numpy.sum(numpy.log(numpy.where(y_train == 1, nextP, 1.0-nextP)))
      # print(step, objFunc1)
      if (not numpy.isnan(objFunc1)):
         if (objFunc1 > objFunc):
            qGoodStep = 1
            step1 = step
            objFunc = objFunc1

   if (qGoodStep == 1):     
      nextP = currentP + step1 * treePredProb
      nextP = numpy.where(nextP >= 1.0, 1.0, nextP)
      nextP = numpy.where(nextP <= 0.0, 0.0, nextP)
      objFunc = numpy.sum(numpy.log(numpy.where(y_train == 1, nextP, 1.0-nextP)))
      print('Step = ', step1)
      print('Objective Function = ', objFunc)

      currentP = nextP
      print('Predicted Event Probability:\n', currentP)
      yPredClass = numpy.where(currentP >= threshPredProb, 1, 0)
      y_accuracy = metrics.accuracy_score(y_train, yPredClass)
      print("Accuracy Score = ", y_accuracy)
