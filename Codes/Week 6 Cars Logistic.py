import numpy
import pandas

import sklearn.metrics as metrics
import statsmodels.api as stats

# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pandas.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100

   print("Frequency Table: \n")
   print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

   return

cars = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv', delimiter=',')

nObs = cars.shape[0]

# Specify Origin as a categorical variable
Origin = cars['Origin'].astype('category')
y = Origin
y_category = y.cat.categories

# Forward Selection   
# Model 0 is Origin = Intercept
X = numpy.where(Origin.notnull(), 1, 0)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))


DriveTrain = cars[['DriveTrain']].astype('category')
X = pandas.get_dummies(DriveTrain)
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

y_confusion = metrics.confusion_matrix(y, y_predictClass)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predictClass)
print("Accuracy Score = ", y_accuracy)

# Model is Origin = Intercept + DriveTrain + EngineSize + Horsepower + Length + Weight.

Origin = cars['Origin'].astype('category')
y = Origin
y_category = y.cat.categories

DriveTrain = cars[['DriveTrain']].astype('category')
X = pandas.get_dummies(DriveTrain)
X = X.join(cars[['EngineSize', 'Horsepower', 'Length', 'Weight']])
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

y_confusion = metrics.confusion_matrix(y, y_predictClass)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predictClass)
print("Accuracy Score = ", y_accuracy)
