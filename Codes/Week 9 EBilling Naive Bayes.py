import numpy
import pandas

from sklearn import preprocessing, naive_bayes

# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

inputData = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\CustomerSurveyData.csv',
                         delimiter=',')

# EBilling -> CreditCard, Gender, JobCategory
subData = inputData[['CreditCard', 'Gender', 'JobCategory', 'EBilling']].dropna()

catEBilling = subData['EBilling'].unique()
catCreditCard = subData['CreditCard'].unique()
catGender = subData['Gender'].unique()
catJobCategory = subData['JobCategory'].unique()

print('Unique Values of EBilling: \n', catEBilling)
print('Unique Values of CreditCard: \n', catCreditCard)
print('Unique Values of Gender: \n', catGender)
print('Unique Values of JobCategory: \n', catJobCategory)

RowWithColumn(rowVar = subData['EBilling'], columnVar = subData['CreditCard'], show = 'ROW')
RowWithColumn(rowVar = subData['EBilling'], columnVar = subData['Gender'], show = 'ROW')
RowWithColumn(rowVar = subData['EBilling'], columnVar = subData['JobCategory'], show = 'ROW')

subData = subData.astype('category')
xTrain = pandas.get_dummies(subData[['CreditCard', 'Gender', 'JobCategory']])

yTrain = numpy.where(subData['EBilling'] == 'Yes', 1, 0)

# Incorrectly Use sklearn.naive_bayes.BernoulliNB

_objNB = naive_bayes.BernoulliNB(alpha = 1e-10)
thisModel = _objNB.fit(xTrain, yTrain)

print('Probability of each class:')
print(numpy.exp(_objNB.class_log_prior_))
print('\n')

print('Empirical probability of features given a class, P(x_i|y)')
print(xTrain.columns)
print(numpy.exp(_objNB.feature_log_prob_))
print('\n')

print('Number of samples encountered for each class during fitting')
print(_objNB.class_count_)
print('\n')

print('Number of samples encountered for each (class, feature) during fitting')
print(_objNB.feature_count_)
print('\n')

xTest = pandas.DataFrame(numpy.zeros((1, xTrain.shape[1])), columns = xTrain.columns)

xTest[['CreditCard_American Express', 'Gender_Female', 'JobCategory_Professional']] = [1,1,1]
y_predProb = thisModel.predict_proba(xTest)

print(y_predProb)

# Correctly Use sklearn.naive_bayes.CategoricalNB
feature = ['CreditCard', 'Gender', 'JobCategory']

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(subData['EBilling'])
yLabel = labelEnc.inverse_transform([0, 1])

uCreditCard = numpy.unique(subData['CreditCard'])
uGender = numpy.unique(subData['Gender'])
uJobCategory = numpy.unique(subData['JobCategory'])

featureCategory = [uCreditCard, uGender, uJobCategory]
print(featureCategory)

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(subData[['CreditCard', 'Gender', 'JobCategory']])

_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(xTrain, yTrain)

print('Number of samples encountered for each class during fitting')
print(yLabel)
print(_objNB.class_count_)
print('\n')

print('Probability of each class:')
print(yLabel)
print(numpy.exp(_objNB.class_log_prior_))
print('\n')

print('Number of samples encountered for each (class, feature) during fitting')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(_objNB.category_count_[i])
   print('\n')

print('Empirical probability of features given a class, P(x_i|y)')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(numpy.exp(_objNB.feature_log_prob_[i]))
   print('\n')

# CreditCard = American Express, Gender = Female, JobCategory = Professional
xTest = featureEnc.transform([['American Express', 'Female', 'Professional']])

y_predProb = thisModel.predict_proba(xTest)
print('Predicted Probability: ', yLabel, y_predProb)
