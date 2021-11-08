import numpy
import pandas
import statsmodels.api as smodel
import time

from itertools import combinations 

# Set some options for printing all the columns
pandas.set_option('precision', 7)

cars = pandas.read_csv('C:\\IIT\Machine Learning\\Data\\cars.csv', delimiter=',')

def SWEEPOperator (pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    
    A = numpy.copy(inputM)
    diagA = numpy.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = numpy.zeros(pDim)
            ANext[k, :] = numpy.zeros(pDim)
        A = ANext
    return (A, aliasParam, nonAliasParam)

inputData = pandas.read_csv('C:\\IIT\Machine Learning\\\Data\\cars.csv',
                            delimiter = ',', header = 0)

catFeature = ['DriveTrain', 'Type', 'Cylinders']
contFeature = ['Horsepower', 'MPG_City', 'MPG_Highway', 'Weight', 'Wheelbase', 'Length']
allFeature = catFeature + contFeature

catTarget = 'Origin'

contMean = inputData[[catTarget] + contFeature].groupby(catTarget).mean()

allCombResult = pandas.DataFrame()

allComb = []
for r in range(len(allFeature)+1):
   allComb = allComb + list(combinations(allFeature, r))

startTime = time.time()

nComb = len(allComb)
for r in range(nComb):
   modelTerm = list(allComb[r])
   trainData = inputData[[catTarget] + modelTerm].dropna()
   Y = trainData[catTarget].astype('category')

   fullX = smodel.add_constant(trainData, prepend = True)
   fullX = fullX[['const']]
   for pred in modelTerm:
      if (pred in catFeature):
         fullX = fullX.join(pandas.get_dummies(trainData[pred].astype('category')))
      elif (pred in contFeature):
         fullX = fullX.join(trainData[pred])

   XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
   pDim = XtX.shape[0]

   invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)

   # The number of free parameters
   modelX = fullX.iloc[:, list(nonAliasParam)]
   objLogit = smodel.MNLogit(Y, modelX)
   thisFit = objLogit.fit(method = 'ncg', maxiter = 200, tol = 1e-8)

   MDF = (thisFit.J - 1) * thisFit.K
   LLK = thisFit.llf

   NSample = len(Y)
   AIC = 2.0 * MDF - 2.0 * LLK
   BIC = MDF * numpy.log(NSample) - 2.0 * LLK
   allCombResult = allCombResult.append([[r, modelTerm, len(modelTerm), LLK, MDF, AIC, BIC, NSample]],
                                          ignore_index = True)
   del objLogit

endTime = time.time()
            
allCombResult = allCombResult.rename(
   columns = {0: 'Step', 1: 'Model Term', 2: 'Number of Terms',
              3: 'Log-Likelihood', 4: 'Model Degree of Freedom',
              5: 'Akaike Information Criterion', 6: 'Bayesian Information Criterion',
              7: 'Sample Size'})

elapsedTime = endTime - startTime
