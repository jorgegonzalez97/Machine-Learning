import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as stats
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.svm as svm

hmeq = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\HMEQ.csv', delimiter=',')

y_name = 'BAD'

# Set dual = False because n_samples > n_features

# Step 1
accuracyResult = pandas.DataFrame()
includeVar = []
X_name = ['CLAGE','CLNO','DELINQ','DEROG','NINQ','YOJ']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult = accuracyResult.append([[includeVar + [ivar], inputData.shape[0], y_predictAccuracy]], ignore_index = True)

# Step 2
accuracyResult = pandas.DataFrame()
includeVar = ['YOJ']
X_name = ['CLAGE','CLNO','DELINQ','DEROG','NINQ']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult = accuracyResult.append([[includeVar + [ivar], inputData.shape[0], y_predictAccuracy]], ignore_index = True)

# Step 3
accuracyResult = pandas.DataFrame()
includeVar = ['YOJ', 'NINQ']
X_name = ['CLAGE','CLNO','DELINQ','DEROG']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult = accuracyResult.append([[includeVar + [ivar], inputData.shape[0], y_predictAccuracy]], ignore_index = True)

# Step 4
accuracyResult = pandas.DataFrame()
includeVar = ['YOJ', 'NINQ', 'CLNO']
X_name = ['CLAGE','DELINQ','DEROG']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult = accuracyResult.append([[includeVar + [ivar], inputData.shape[0], y_predictAccuracy]], ignore_index = True)

# Step 5
accuracyResult = pandas.DataFrame()
includeVar = ['YOJ', 'NINQ', 'CLNO', 'CLAGE']
X_name = ['DELINQ','DEROG']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult = accuracyResult.append([[includeVar + [ivar], inputData.shape[0], y_predictAccuracy]], ignore_index = True)

# Step 6
accuracyResult = pandas.DataFrame()
includeVar = ['YOJ', 'NINQ', 'CLNO', 'CLAGE', 'DEROG']
X_name = ['DELINQ']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult = accuracyResult.append([[includeVar + [ivar], inputData.shape[0], y_predictAccuracy]], ignore_index = True)
