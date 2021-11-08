import graphviz
import matplotlib.pyplot as plt
import numpy
import sklearn.tree as tree

x_train = numpy.array([[0.1, 0.3],
                       [0.2, 0.2],
                       [0.3, 0.1],
                       [0.4, 0.4],
                       [0.5, 0.7],
                       [0.6, 0.5],
                       [0.7, 0.9],
                       [0.8, 0.8],
                       [0.8, 0.2],
                       [0.9, 0.8]], dtype = float)

y_train = numpy.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype = float)

carray = ['blue', 'red']
plt.figure(figsize=(16,9))
i0 = numpy.where(y_train == 0)
plt.scatter(x_train[i0,0], x_train[i0,1], c = carray[0], label = 0, s = 100)
i1 = numpy.where(y_train == 1)
plt.scatter(x_train[i1,0], x_train[i1,1], c = carray[1], label = 1, s = 100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.axis(aspect = 'equal')
plt.legend(title = 'y', fontsize = 12, markerscale = 1)
plt.show()

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60616)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
print('Accuracy = ', accuracy)

dot_data = tree.export_graphviz(treeFit,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['X1', 'X2'],
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
graph

carray = ['blue', 'red']
plt.figure(figsize=(16,9))
i0 = numpy.where(y_train == 0)
plt.scatter(x_train[i0,0], x_train[i0,1], c = carray[0], label = 0, s = 100)
i1 = numpy.where(y_train == 1)
plt.scatter(x_train[i1,0], x_train[i1,1], c = carray[1], label = 1, s = 100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot([0.25, 0.25], [0, 1], color = 'red', linestyle = ':')
plt.plot([0.85, 0.85], [0, 1], color = 'red', linestyle = ':')
plt.plot([0, 1], [0.6, 0.6], color = 'red', linestyle = ':')
plt.grid(True)
plt.axis(aspect = 'equal')
plt.legend(title = 'y', fontsize = 12, markerscale = 1)
plt.show()

# Build a classification tree on the training partition
w_train = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype = float)

for iter in range(4):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=60616)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    y_train_predclass = numpy.where(treePredProb[:,1] >= 0.5, 1, 0)
    accuracy = numpy.sum(numpy.where(y_train == y_train_predclass, w_train, 0.0)) / numpy.sum(w_train)

    accuracy2 = classTree.score(x_train, y_train, w_train)
    print('Accuracy = ', accuracy, accuracy2)

    if (abs(1.0 - accuracy) < 0.0000001):
        break
    
    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - treePredProb[:,1]), (0 - treePredProb[:,1]))
    predClass = numpy.where(treePredProb[:,1] >= 0.5, 1, 0)
    w_train = numpy.where(predClass != y_train, 1+numpy.abs(eventError), numpy.abs(eventError))

    print('Event Error:\n', eventError)
    print('Predicted Class:\n', predClass)
    print('Weight:\n', w_train)   

