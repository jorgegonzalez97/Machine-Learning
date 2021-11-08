# Load the necessary libraries
import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.tree as tree

trainData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\hmeq.csv',
                             delimiter=',', usecols = ['BAD', 'DELINQ'])

# Remove all missing observations
trainData = trainData.dropna()

# Examine a portion of the data frame
print(trainData)

# Put the descriptive statistics into another dataframe
trainData_descriptive = trainData.describe()
print(trainData_descriptive)

# Horizontal frequency bar chart of BAD
trainData.groupby('BAD').size().plot(kind='barh')
plt.title("Barchart of BAD")
plt.xlabel("Number of Observations")
plt.ylabel("BAD")
plt.grid(axis="x")
plt.show()

# Visualize the histogram of the DELINQ variable
trainData.hist(column='DELINQ', bins=15)
plt.title("Histogram of DELINQ")
plt.xlabel("DELINQ")
plt.ylabel("Number of Observations")
plt.xticks(numpy.arange(0,15,step=1))
plt.grid(axis="x")
plt.show()

# Visualize the boxplot of the DELINQ variable by BAD
trainData.boxplot(column='DELINQ', by='BAD', vert=False)
plt.title("Boxplot of DELINQ by Levels of BAD")
plt.suptitle("")
plt.xlabel("DELINQ")
plt.ylabel("BAD")
plt.grid(axis="y")
plt.show()

X_inputs = trainData[['DELINQ']]
Y_target = trainData[['BAD']]

# Load the TREE library from SKLEARN
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)

hmeq_dt = classTree.fit(X_inputs, Y_target)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(classTree.score(X_inputs, Y_target)))

dot_data = tree.export_graphviz(hmeq_dt,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['DELINQ'],
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)

graph

graph.render('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\hmeq_output',
             format = 'png')

scoreData = trainData.copy()
scoreData['predProb'] = 0.9222
scoreData.loc[scoreData['DELINQ'] == 0, 'predProb'] = 0.1395
scoreData.loc[scoreData['DELINQ'] == 1, 'predProb'] = 0.4059
scoreData.loc[scoreData['DELINQ'] == 2, 'predProb'] = 0.4059
scoreData.loc[scoreData['DELINQ'] == 3, 'predProb'] = 0.4059
scoreData.loc[scoreData['DELINQ'] == 4, 'predProb'] = 0.4059

scoreData['predClass'] = 0
scoreData.loc[scoreData['predProb'] >= 0.4059, 'predClass'] = 1

scoreData['MisClassify'] = 0
scoreData.loc[scoreData['predClass'] != scoreData['BAD'], 'MisClassify'] = 1

print('Misclassification Rate: {:.2%}'.format(scoreData['MisClassify'].mean()))