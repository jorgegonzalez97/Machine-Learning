import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.ensemble as ensemble
import sklearn.tree as tree

trainData = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\SpiralWithCluster.csv', delimiter=',')

nObs = trainData.shape[0]

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

y_threshold = numpy.mean(trainData['SpectralCluster'])

x_train = trainData[['x','y']]
y_train = trainData['SpectralCluster']

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60616)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
print('Accuracy = ', accuracy)

dot_data = tree.export_graphviz(treeFit,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['x', 'y'],
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
graph

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)

plt.plot([-4.5, 4.5], [-2.248, -2.248], color = 'black', linestyle = ':')
plt.plot([-4.5, 4.5], [2.121, 2.121], color = 'black', linestyle = ':')
plt.plot([-4.5, 4.5], [0.346, 0.346], color = 'black', linestyle = ':')
plt.plot([-1.874, -1.874], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([1.88, 1.88], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([3.042, 3.042], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([-3.131, -3.131], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([-4.5, 4.5], [-0.101, -0.101], color = 'black', linestyle = ':')
plt.plot([1.986, 1.986], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([-0.114, -0.114], [-3.5, 3.5], color = 'black', linestyle = ':')

plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

# Build a classification tree on the training partition
w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(28)
ensemblePredProb = numpy.zeros((nObs, 2))

for iter in range(28):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy[iter] = classTree.score(x_train, y_train, w_train)
    ensemblePredProb += accuracy[iter] * treePredProb

    if (abs(1.0 - accuracy[iter]) < 0.0000001):
        break
    
    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - treePredProb[:,1]), (0 - treePredProb[:,1]))
    predClass = numpy.where(treePredProb[:,1] >= 0.5, 1, 0)
    w_train = numpy.where(predClass != y_train, 1+numpy.abs(eventError), numpy.abs(eventError))
 
# Calculate the final predicted probabilities
ensemblePredProb /= numpy.sum(accuracy)

trainData['predCluster'] = numpy.where(ensemblePredProb[:,1] >= 0.5, 1, 0)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['predCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Cluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
boostTree = ensemble.AdaBoostClassifier(base_estimator=classTree, n_estimators=28,
                                        learning_rate=1.0, algorithm='SAMME.R', random_state=None)
boostFit = boostTree.fit(x_train, y_train)
boostPredProb = boostFit.predict_proba(x_train)
boostAccuracy = boostFit.score(x_train, y_train)

trainData['predCluster'] = numpy.where(boostPredProb[:,1] >= 0.5, 1, 0)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['predCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Cluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()
