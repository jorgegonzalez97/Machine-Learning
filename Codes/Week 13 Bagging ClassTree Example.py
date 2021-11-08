import matplotlib.pyplot as plt
import numpy
import pandas
import random
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import sklearn.tree as tree

# Create a bootstrap sample from the population
def sample_wr (inData):
    n = len(inData)
    outData = numpy.empty((n,1))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

# Create the model metrics
def ModelMetrics (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Threshold):      # The probability threshold for declaring a predicted event

    # The Area Under Curve metric
    AUC = metrics.roc_auc_score(DepVar, EventPredProb)

    # The Root Average Squared Error and the Misclassification Rate
    nObs = len(DepVar)
    RASE = 0
    MisClassRate = 0
    for i in range(nObs):
        p = EventPredProb[i]
        if (DepVar[i] == EventValue):
            RASE += (1.0 - p)**2
            if (p < Threshold):
                MisClassRate += 1
        else:
            RASE += p**2
            if (p >= Threshold):
                MisClassRate += 1
    RASE = numpy.sqrt(RASE / nObs)
    MisClassRate /= nObs

    return(AUC, RASE, MisClassRate)

hmeq = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\hmeq.csv', delimiter=',', usecols = ['DELINQ', 'DEBTINC', 'BAD'])
hmeq = hmeq.dropna()

# Partition the data, 70% for training and 30% for testing
x_train, x_test, y_train, y_test = model_selection.train_test_split(hmeq[['DELINQ', 'DEBTINC']], hmeq[['BAD']], test_size = 0.3,
                                                                    random_state = 20191113, stratify = hmeq[['BAD']])

# Calculate the threshold value for declaring a predicted event
threshold = len(y_train[y_train['BAD'] == 1]) / len(y_train)
print('Threshold Value for Declaring a Predicted Event = {:.7f}' .format(threshold))

# Build a classification tree on the training partition
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
treeFit = classTree.fit(x_train, y_train['BAD'])
treePredProb = classTree.predict_proba(x_test)
AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

nB = 0

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('        Without Bootstraps: ')
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

# Build a classification tree for the bootstrap samples
def bootstrap_classTree (x_train, y_train, x_test, nB):
   x_index = x_train.index
   nT = len(x_test)
   outProb = numpy.zeros((nT,2))
   outThreshold = numpy.zeros((nB, 1))
   classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)

   # Initialize internal state of the random number generator.
   random.seed(20191113)

   for iB in range(nB):
      bootIndex = sample_wr(x_index)
      x_train_boot = x_train.loc[bootIndex[:,0]]
      y_train_boot = y_train.loc[bootIndex[:,0]]
      outThreshold[iB] = len(y_train_boot[y_train_boot['BAD'] == 1]) / len(y_train_boot)
      treeFit = classTree.fit(x_train_boot, y_train_boot['BAD'])
      outProb = outProb + classTree.predict_proba(x_test)
   outProb = outProb / nB
   print('Mean Threshold: {:.7f}' .format(outThreshold.mean()))
   print('  SD Threshold: {:.7f}' .format(outThreshold.std()))
   return outProb

nB = 10
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

nB = 50
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

nB = 100
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

nB = 500
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

nB = 1000
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

nB = 5000
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

nB = 10000
treePredProb = bootstrap_classTree (x_train, y_train, x_test, nB)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, treePredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

# Replicate results using the sklearn.ensemble.BaggingClassifier() function

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
bagTree = ensemble.BaggingClassifier(base_estimator = classTree, n_estimators = nB, random_state = 20191113, verbose = 1)
bagFit = bagTree.fit(x_train, y_train['BAD'])
bagPredProb = bagFit.predict_proba(x_test)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_test['BAD'].values, 1, bagPredProb[:,1], threshold)

y_test['P_BAD1'] = treePredProb[:,1]
y_test.boxplot(column='P_BAD1', by='BAD', vert = False, figsize=(6,4))
plt.title("Boxplot of P_BAD1 by Levels of BAD (nb = " + str(nB) + ')')
plt.suptitle(" ")
plt.xlabel("P_BAD1")
plt.ylabel("BAD")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

