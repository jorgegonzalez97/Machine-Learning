# Name: Cars Logistic Forward Selection.py
# Creation Date: October 6, 2020
# Author: Ming-Long Lam

import matplotlib.pyplot as plt
import numpy
import pandas
import scipy
import statsmodels.api as stats

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# Set some options for printing all the columns
pandas.set_option('precision', 7)

inputData = pandas.read_csv('C:\\IIT\Machine Learning\\Data\\cars.csv',
                            delimiter=',', usecols = ['Origin', 'DriveTrain', 'Weight'])

trainData = inputData.dropna()

# Rename the columns
yName = 'Origin'
catName = 'DriveTrain'
intName = 'Weight'

# Frequency of the categorical fields
print('\n--- Frequency of ' + yName + ' ---')
print(trainData[yName].value_counts())

print('\n--- Frequency of ' + catName + ' ---')
print(trainData[catName].value_counts())
    
# Descriptive statistics of the interval field
print('\n--- Descriptive Statistics of ' + intName + ' ---')
print(trainData[intName].describe())

# Specify the color sequence
cmap = ['indianred','sandybrown','royalblue']

# Generate the contingency table of the categorical input feature by the target
cntTable = pandas.crosstab(index = trainData[catName], columns = trainData[yName],
                           margins = False, dropna = True)

# Calculate the row percents
pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')

# Generate a horizontal stacked percentage bar chart
barThick = 0.8
yCat = cntTable.columns
accPct = numpy.zeros(pctTable.shape[0])
fig, ax = plt.subplots()
for j in range(len(yCat)):
    catLabel = yCat[j]
    plt.barh(pctTable.index, pctTable[catLabel], color = cmap[j], left = accPct, label = catLabel,
             height = barThick)
    accPct = accPct + pctTable[catLabel]
ax.xaxis.set_major_locator(MultipleLocator(base = 20))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax.xaxis.set_minor_locator(MultipleLocator(base = 5))
ax.set_xlabel('Percent')
ax.set_ylabel(catName)
plt.grid(axis = 'x')
plt.legend(loc = 'lower center', bbox_to_anchor = (0.35, 1), ncol = 3)
plt.show()

# Generate the contingency table of the interval input feature by the target
cntTable = pandas.crosstab(index = trainData[intName], columns = trainData[yName],
                           margins = False, dropna = True)

# Calculate the row percents
pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')
yCat = cntTable.columns
fig, ax = plt.subplots()
plt.stackplot(pctTable.index, numpy.transpose(pctTable), baseline = 'zero', colors = cmap, labels = yCat)
ax.xaxis.set_major_locator(MultipleLocator(base = 1000))
ax.xaxis.set_minor_locator(MultipleLocator(base = 200))
ax.yaxis.set_major_locator(MultipleLocator(base = 20))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax.yaxis.set_minor_locator(MultipleLocator(base = 5))
ax.set_xlabel(intName)
ax.set_ylabel('Percent')
plt.grid(axis = 'both')
plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1), ncol = 3)
plt.show()

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
            ANext[:, k] = 0.0 * A[:, k]
            ANext[k, :] = ANext[:, k]
        A = ANext
    return (A, aliasParam, nonAliasParam)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = numpy.transpose(fullX).dot(fullX)
    invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-7)

    # Build a multinomial logistic model
    X = fullX.iloc[:, list(nonAliasParam)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method = 'newton', maxiter = 1000, gtol = 1e-6, full_output = True, disp = True)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    # The number of free parameters
    nYCat = thisFit.J
    thisDF = len(nonAliasParam) * (nYCat - 1)

    # Return model statistics
    return (thisLLK, thisDF, thisParameter, thisFit)

# Reorder the categories in ascending order of frequencies
# Create dummy indicators for the categorical input feature
catFreq = trainData[catName].value_counts()
catFreq = catFreq.sort_values(ascending = True)
newCat = catFreq.index
u = trainData[catName].astype('category')
xCat = pandas.get_dummies(u.cat.reorder_categories(newCat))

# Column for the interval input feature
xInt = trainData[[intName]]

# Reorder the categories in descending order of frequencies of the target field
catFreq = trainData[yName].value_counts()
catFreq = catFreq.sort_values(ascending = False)
newCat = catFreq.index
u = trainData[yName].astype('category')
y = u.cat.reorder_categories(newCat)
print('Target Categories:\n', y.cat.categories)

# Train a Logistic Regression model using the Forward Selection method
devianceTable = pandas.DataFrame()

u = pandas.DataFrame()

# Step 0: Intercept only model
u = y.isnull()
designX = pandas.DataFrame(u.where(u, 1)).rename(columns = {yName: "const"})
LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)
devianceTable = devianceTable.append([[0, 'Intercept', DF0, LLK0, None, None, None]])

# Step 1.1: Intercept + 'DriveTrain'
designX = xCat
designX = stats.add_constant(designX, prepend = True)
LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
testDev = 2.0 * (LLK1 - LLK0)
testDF = DF1 - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
devianceTable = devianceTable.append([[1.1, 'Intercept + DriveTrain',
                                       DF1, LLK1, testDev, testDF, testPValue]])

# Step 1.2: Intercept + 'Weight'
designX = xInt
designX = stats.add_constant(designX, prepend = True)
LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
testDev = 2.0 * (LLK1 - LLK0)
testDF = DF1 - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
devianceTable = devianceTable.append([[1.2, 'Intercept + Weight', DF1, LLK1, testDev, testDF, testPValue]])

# Step 1: Intercept + 'DriveTrain'
designX = xCat
designX = stats.add_constant(designX, prepend = True)
LLK0, DF0, fullParams1, thisFit = build_mnlogit (designX, y)
devianceTable = devianceTable.append([[1, 'Intercept + DriveTrain', DF0, LLK0, None, None, None]])

# Step 2.1: Intercept + 'DriveTrain' + 'Weight'
designX = xInt
designX = designX.join(xCat)
designX = stats.add_constant(designX, prepend = True)

LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
testDev = 2.0 * (LLK1 - LLK0)
testDF = DF1 - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
devianceTable = devianceTable.append([[2.1, 'Intercept + Weight + DriveTrain', DF1, LLK1, testDev, testDF, testPValue]])

devianceTable = devianceTable.rename(columns = {0:'Sequence', 1:'Model Specification',
                                                2:'Number of Free Parameters', 3:'Log-Likelihood',
                                                4:'Deviance', 5:'Degree of Freedom', 6:'Significance'})
