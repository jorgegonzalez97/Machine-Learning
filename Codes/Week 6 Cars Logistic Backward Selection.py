import numpy
import pandas
import scipy

import statsmodels.api as stats

cars = pandas.read_csv('C:\\MScAnalytics\\Data Mining Principles\\Data\\cars.csv',
                       delimiter=',', usecols = ['Origin', 'DriveTrain', 'Weight'])

nObs = cars.shape[0]

# Specify Origin as a categorical variable
Origin = cars['Origin'].astype('category')
y = Origin
y_category = y.cat.categories

# Backward Selection
# Consider Model 0 is Origin = Intercept + DriveTrain + Weight
DriveTrain = cars[['DriveTrain']].astype('category')
X = pandas.get_dummies(DriveTrain)
X = X.join(cars[['Weight']])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)

# Consider Model 1 is Origin = Intercept + DriveTrain
DriveTrain = cars[['DriveTrain']].astype('category')
X = pandas.get_dummies(DriveTrain)
X = stats.add_constant(X, prepend=True)
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 1 is Origin = Intercept + Weight
X = cars[['Weight']]
X = stats.add_constant(X, prepend=True)
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)
