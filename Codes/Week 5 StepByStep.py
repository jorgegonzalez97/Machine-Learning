# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy
import pandas

# Define a function to visualize the percent of a particular target category by an interval predictor
def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   dataTable = inData
   dataTable['LE_Split'] = (dataTable.iloc[:,0] <= split)

   crossTable = pandas.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * numpy.log2(proportion)
      print('Row = ', iRow, 'Entropy =', rowEntropy)
      print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
   return(tableEntropy)

cars = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv',
                       delimiter=',')

inData2 = cars[['Cylinders', 'Origin']].dropna()

# Horizontal frequency bar chart of Cylinders
inData2.groupby('Cylinders').size().plot(kind='barh')

# Horizontal frequency bar chart of Cylinders
inData2.groupby('Origin').size().plot(kind='barh')

crossTable = pandas.crosstab(index = inData2['Cylinders'], columns = inData2['Origin'],
                             margins = True, dropna = True)   
print(crossTable)

# Split (3), (4, 5, 6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 2.5)
print('Split Entropy = ', EV)

# Split (3), (4, 5, 6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 3.5)
print('Split Entropy = ', EV)

# Split (3, 4), (5, 6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 4.5)
print('Split Entropy = ', EV)

# Split (3, 4, 5), (6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 5.5)
print('Split Entropy = ', EV)

# Split (3, 4, 5, 6), (8, 10, 12)
EV = EntropyIntervalSplit(inData2, 7)
print('Split Entropy = ', EV)

# Split (3, 4, 5, 6, 8), (10, 12)
EV = EntropyIntervalSplit(inData2, 9)
print('Split Entropy = ', EV)

# Split (3, 4, 5, 6, 8, 10), (12)
EV = EntropyIntervalSplit(inData2, 11)
print('Split Entropy = ', EV)

LeftBranch = inData2[inData2['Cylinders'] <= 7.0]

# Split (3), (4, 5, 6)
EV = EntropyIntervalSplit(LeftBranch, 3.5)
print('Split Entropy = ', EV)

# Split (3, 4), (5, 6)
EV = EntropyIntervalSplit(LeftBranch, 4.5)
print('Split Entropy = ', EV)

# Split (3, 4, 5), (6)
EV = EntropyIntervalSplit(LeftBranch, 5.5)
print('Split Entropy = ', EV)

RightBranch = inData2[inData2['Cylinders'] > 7.0]

# Split (8), (10, 12)
EV = EntropyIntervalSplit(RightBranch, 9)
print('Split Entropy = ', EV)

# Split (8, 10), (12)
EV = EntropyIntervalSplit(RightBranch, 11)
print('Split Entropy = ', EV)

# Load the TREE library from SKLEARN
from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
cars_DT = classTree.fit(inData2[['Cylinders']], inData2['Origin'])
DT_accuracy = classTree.score(inData2[['Cylinders']], inData2['Origin'])

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(DT_accuracy))

import graphviz
dot_data = tree.export_graphviz(cars_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['Cylinders'],
                                class_names = ['Asia', 'Europe', 'USA'])

graph = graphviz.Source(dot_data)
graph

graph.render('C:\\IIT\\Machine Learning\\Job\\cars_DT_output')
