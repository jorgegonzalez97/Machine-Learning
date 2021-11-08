# Load the necessary libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy
import pandas

# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominal (
   targetVar,       # target variable
   targetCat,       # target category
   predictor,       # nominal predictor
   val4na):         # imputed value for NaN

   crossTable = pandas.crosstab(index = predictor.fillna(val4na), columns = targetVar, margins = True, dropna = True)
   crossTable['Percent'] = 100 * (crossTable[targetCat] / crossTable['All'])
   print(crossTable)

   plotTable = crossTable[crossTable.index != 'All']
   plt.bar(plotTable.index, plotTable['Percent'])
   plt.xlabel(predictor.name)
   plt.ylabel('Percent of ' + targetVar.name + ' = ' + str(targetCat))
   plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
   plt.grid(True, axis='y')
   plt.show()

   return(crossTable)

# Define a function to visualize the percent of a particular target category by an interval predictor
def TargetPercentByInterval (
   targetVar,       # target variable
   targetCat,       # target category
   predictor,       # nominal predictor
   val4na):         # imputed value for NaN

   crossTable = pandas.crosstab(index = predictor.fillna(val4na), columns = targetVar, margins = True, dropna = True)
   crossTable['Percent'] = 100 * (crossTable[targetCat] / crossTable['All'])
   print(crossTable)

   plotTable = crossTable[crossTable.index != 'All']
   plt.scatter(plotTable.index, plotTable['Percent'])
   plt.xlabel(predictor.name)
   plt.ylabel('Percent of ' + targetVar.name + ' = ' + str(targetCat))
   plt.grid(True, axis='both')
   plt.show()

   return(crossTable)

hmeq = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',')
nTotal = len(hmeq)

# Generate the frequency table and the bar chart for the BAD target variable
crossTable = pandas.crosstab(index = hmeq['BAD'], columns = ["Count"], margins = True, dropna = False)
crossTable['Percent'] = 100 * (crossTable['Count'] / nTotal)
crossTable = crossTable.drop(columns = ['All'])
print(crossTable)

plotTable = crossTable[crossTable.index != 'All']

plt.bar(plotTable.index, plotTable['Percent'])
plt.xticks([[0], [1]])
plt.xlabel('BAD')
plt.ylabel('Percent')
plt.grid(True, axis='y')
plt.show()

# Cross-tabulate BAD by DELINQ
resultTable = TargetPercentByNominal(hmeq['BAD'], 1, hmeq['DELINQ'], val4na = -1)

# Cross-tabulate BAD by DEROG
resultTable = TargetPercentByNominal(hmeq['BAD'], 1, hmeq['DEROG'], val4na = -1)

# Cross-tabulate BAD by JOB
resultTable = TargetPercentByNominal(hmeq['BAD'], 1, hmeq['JOB'], val4na = 'Unknown')

# Cross-tabulate BAD by NINQ
resultTable = TargetPercentByNominal(hmeq['BAD'], 1, hmeq['NINQ'], val4na = -1)

# Cross-tabulate BAD by REASON
resultTable = TargetPercentByNominal(hmeq['BAD'], 1, hmeq['REASON'], val4na = 'Unknown')

# Cross-tabulate BAD by DEBTINC
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['DEBTINC'], val4na = -10)

# Cross-tabulate BAD by CLAGE
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['CLAGE'], val4na = -10)

# Cross-tabulate BAD by CLNO
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['CLNO'], val4na = -10)

# Cross-tabulate BAD by LOAN
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['LOAN'], val4na = -10)

# Cross-tabulate BAD by MORTDUE
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['MORTDUE'], val4na = -10)

# Cross-tabulate BAD by VALUE
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['VALUE'], val4na = -10)

# Cross-tabulate BAD by YOJ
resultTable = TargetPercentByInterval(hmeq['BAD'], 1, hmeq['YOJ'], val4na = -10)

# Specify the target and the predictor variables
X_name = ['DEBTINC', 'DELINQ']
Y_name = 'BAD'

trainData = hmeq[['DEBTINC', 'DELINQ', 'BAD']].dropna()

X_inputs = trainData[X_name]
Y_target = trainData[Y_name]

# How many missing values are there?
print('Number of Missing Observations:')
print(X_inputs.isnull().sum())
print(Y_target.isnull().sum())

# Load the TREE library from SKLEARN
from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)

hmeq_DT = classTree.fit(X_inputs, Y_target)
print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(X_inputs, Y_target)))

import graphviz
dot_data = tree.export_graphviz(hmeq_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_name,
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
print(graph)
