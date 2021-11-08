import numpy
import pandas

import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import statsmodels.api as smodel

# Set some options for printing all the columns
pandas.set_option('precision', 7)

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

# The Home Equity Loan example
catTarget = 'BAD'
intPred = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'CLAGE', 'CLNO', 'DEBTINC']

hmeq = pandas.read_csv('C:\\IIT\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',', usecols = [catTarget]+intPred)
trainData = hmeq.dropna()

Y = trainData[catTarget].astype('category')
fullX = trainData[intPred]
fullX.insert(0, '_Intercept', 1.0)

XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]

invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print(fullX.columns[list(aliasParam)])

modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)

# Logistic regression with L1 regularization
objLogit = linear_model.LogisticRegression(penalty = 'l1', fit_intercept = True, random_state = 31008,
                                           solver = 'liblinear', max_iter = 1000, tol = 1e-4)
thisFit = objLogit.fit(modelX, Y)
intercept_l1 = thisFit.intercept_
param_l1 = pandas.Series(thisFit.coef_[0,:], index = modelX.columns)
predProb = objLogit.predict_proba(modelX)
AUC_l1 = metrics.roc_auc_score(Y, predProb[:,1])

# Logistic regression with L2 regularization
objLogit = linear_model.LogisticRegression(penalty = 'l2', fit_intercept = True,  random_state = 31008,
                                           max_iter = 1000, tol = 1e-4)
thisFit = objLogit.fit(modelX, Y)
intercept_l2 = thisFit.intercept_
param_l2 = pandas.Series(thisFit.coef_[0,:], index = modelX.columns)
predProb = objLogit.predict_proba(modelX)
AUC_l2 = metrics.roc_auc_score(Y, predProb[:,1])

# Logistic regression with L1/L2 regularization
objLogit = linear_model.LogisticRegression(penalty = 'elasticnet', fit_intercept = True,
                                           l1_ratio = 0.5, solver = 'saga', max_iter = 10000,
                                           tol = 1e-4,  random_state = 31008)
thisFit = objLogit.fit(modelX, Y)
intercept_elasticnet = thisFit.intercept_
param_elasticnet = pandas.Series(thisFit.coef_[0,:], index = modelX.columns)
predProb = objLogit.predict_proba(modelX)
AUC_elasticnet = metrics.roc_auc_score(Y, predProb[:,1])

# Logistic regression without any regularization
modelX = smodel.add_constant(modelX, prepend = True)
objLogit = smodel.MNLogit(Y, modelX)
thisFit = objLogit.fit(method = 'ncg', maxiter = 200, tol = 1e-8)
param_none = thisFit.params
pvalue_none = thisFit.pvalues
predProb = thisFit.predict(modelX)
AUC_none = metrics.roc_auc_score(Y, predProb[1])
