import numpy
import pandas
import scipy.stats as sdist
import statsmodels.api as smodel

# Set some options for printing all the columns
pandas.set_option('precision', 7)

# Define a function that performs the Pearson Chi-square test
#   xCat - Input categorical feature (array-like or Series)
#   yCat - Input categorical target field (array-like or Series)

def PearsonChiSquareTest (xCat, yCat):
    # Generate the crosstabulation
    obsCount = pandas.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    xNCat = obsCount.shape[0]
    yNCat = obsCount.shape[1]
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    # Calculate the Chi-Square statistics
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (xNCat - 1) * (yNCat - 1)
    if (chiSqDf > 0):
       chiSqSig = sdist.chi2.sf(chiSqStat, chiSqDf)
    else:
       chiSqSig = numpy.NaN

    return (xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig)

# Define a function that performs the Deviance Chi-square test
#   xCont - Input categorical feature (array-like or Series)
#   yCat - Input categorical target field (array-like or Series)

def DevianceTest (xCont, yCat):
    # Train a model with the Intercept term and xCont
    X = smodel.add_constant(xCont, prepend = True)
    y = yCat.astype('category')
    logit = smodel.MNLogit(y, X)
    yNCat = logit.J
    thisFit = logit.fit(method = 'newton', maxiter = 1000, full_output = False, disp = False)
    chiSqStat = thisFit.llr
    chiSqDf = thisFit.df_model
    chiSqSig = thisFit.llr_pvalue

    return (yNCat, chiSqStat, chiSqDf, chiSqSig)

# Define a function that performs the ANOVA test
#   xCat - Input categorical feature (array-like or Series)
#   yCont - Input continuous target field (array-like or Series)

def AnalysisOfVarianceTest (xCat, yCont):
   df = pandas.DataFrame(columns = ['x', 'y'])
   df['x'] = xCat
   df['y'] = yCont

   # Total Count and Sum of Squares
   totalCount = df['y'].count()
   totalSSQ = df['y'].var(ddof = 0) * totalCount

   # Within Group Count and Sums of Squares
   groupCount = df.groupby('x').count()
   groupSSQ = df.groupby('x').var(ddof = 0) * groupCount
   nGroup = groupCount.shape[0]

   withinSSQ = numpy.sum(groupSSQ.values)
   betweenSSQ = max(0.0, (totalSSQ - withinSSQ))

   # Compute F statistics
   fDf1 = (nGroup - 1)
   fDf2 = (totalCount - nGroup)
   if (fDf1 > 0 and fDf2 > 0 and withinSSQ > 0.0):
      fStat = (betweenSSQ / fDf1) / (withinSSQ / fDf2)
      fSig = sdist.f.sf(fStat, fDf1, fDf2)
   else:
      fStat = numpy.NaN
      fSig = numpy.NaN

   xNCat = nGroup
   return (xNCat, fStat, fDf1, fDf2, fSig)

# Define a function that performs the Regression test
#   xCont - Input continuous feature (array-like or Series)
#   yCont - Input continuous target field (array-like or Series)

def RegressionTest (xCont, yCont):
   nObs = len(yCont)
   xyCov = numpy.cov(xCont, yCont, ddof = 0)
   tDf = nObs - 2
   tStat = numpy.NaN
   tSig = numpy.NaN
   if (tDf > 0 and xyCov[0,0] > 0.0):
      xMean = numpy.mean(xCont)
      yMean = numpy.mean(yCont)
      regB = xyCov[0,1] / xyCov[0,0]
      yHat = yMean + regB * (xCont - xMean)
      residVariance = numpy.sum((yCont - yHat)**2) / tDf
      if (residVariance > 0.0):
         seB = numpy.sqrt(residVariance / (nObs * xyCov[0,0]))
         tStat = regB / seB
         tSig = 2.0 * sdist.t.sf(abs(tStat), tDf)

   return (tStat, tDf, tSig)

# The Home Equity Loan example
catPred = ['REASON', 'JOB', 'DEROG', 'DELINQ', 'NINQ']
intPred = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'CLAGE', 'CLNO', 'DEBTINC']

hmeq = pandas.read_csv('C:\\IIT\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',', usecols = ['BAD']+catPred+intPred)
hmeq = hmeq.dropna()

testResult = pandas.DataFrame()

for pred in catPred:
    xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig = PearsonChiSquareTest(hmeq[pred], hmeq['BAD'])
    testResult = testResult.append([[pred, xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig]], ignore_index = True)
    
for pred in intPred:
    yNCat, chiSqStat, chiSqDf, chiSqSig = DevianceTest(hmeq[pred], hmeq['BAD'])
    testResult = testResult.append([[pred, numpy.NaN, yNCat, chiSqStat, chiSqDf, chiSqSig]], ignore_index = True)

testResult = testResult.rename(columns = {0:'Feature', 1: 'Feature N Category', 2: 'Target N Category',
                                          3:'Statistic', 4:'DF', 5:'Significance'})
rankSig = testResult.sort_values('Significance', axis = 0, ascending = True)
print(rankSig)
