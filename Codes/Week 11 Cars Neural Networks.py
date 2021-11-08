import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.neural_network as nn

inputData = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv', delimiter=',')

target = 'DriveTrain'

catPred = ['Type','Origin','Cylinders']
intPred = ['EngineSize','Horsepower','MPG_City','MPG_Highway','Weight','Wheelbase','Length']

inputData[catPred] = inputData[catPred].astype('category')
X = pandas.get_dummies(inputData[catPred].astype('category'))
X = X.join(inputData[intPred])

y = inputData[target].astype('category')
y_category = y.cat.categories
y_dummy = pandas.get_dummies(y).to_numpy(dtype = float)

def Build_NN_Class (actFunc, nLayer, nHiddenNeuron):

   # Build Neural Network
   nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = actFunc, verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 10000, random_state = 20201104)

   thisFit = nnObj.fit(X, y)
   y_predProb = nnObj.predict_proba(X)

   # Calculate Root Average Squared Error
   y_residual = y_dummy - y_predProb
   rase = numpy.sqrt(numpy.mean(y_residual ** 2))
   return (rase)

result = pandas.DataFrame(columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', 'RASE'])

for i in numpy.arange(1,11):
    for j in numpy.arange(5,25,5):
        for act in ['identity','logistic','relu','tanh']:
           RASE = Build_NN_Class (actFunc = act, nLayer = i, nHiddenNeuron = j)
           result = result.append(pandas.DataFrame([[act, i, j, RASE]], 
                                  columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', 'RASE']),
                                  ignore_index=True)
           
result[['Activation Function','RASE']].boxplot(by = 'Activation Function')
plt.suptitle('')
plt.title('')
plt.xlabel('Activation Function')
plt.ylabel('Root Average Squared Error')
plt.show()

result[['nLayer','RASE']].boxplot(by = 'nLayer')
plt.suptitle('')
plt.title('')
plt.xlabel('Number of Layers')
plt.ylabel('Root Average Squared Error')
plt.show()

result[['nHiddenNeuron','RASE']].boxplot(by = 'nHiddenNeuron')
plt.suptitle('')
plt.title('')
plt.xlabel('Number of Hidden Neurons per Layer')
plt.ylabel('Root Average Squared Error')
plt.show()

# Train this NN: RELU, 1-layer, 10-neurons each
actFunc = 'relu'
nLayer = 1
nHiddenNeuron = 10
nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                         activation = actFunc, verbose = False,
                         solver = 'lbfgs', learning_rate_init = 0.1,
                         max_iter = 10000, random_state = 20201104)

thisFit = nnObj.fit(X, y)
y_predProb = nnObj.predict_proba(X)

nBin = 10
maxY = 8
thisIndex = y[y == 'AWD'].index
thisPredProb = y_predProb[thisIndex,:]
ax1 = plt.subplot(3,3,1)
counts, bins, patches = plt.hist(thisPredProb[:,0], nBin, density=True, facecolor='r', alpha=0.75)
ax1.set_xlim(0.0,1.0)
ax1.set_ylim(0.0,maxY)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_title('Pred.Prob.: AWD')
ax1.set_ylabel('Obs.: AWD')

ax2 = plt.subplot(3,3,2)
counts, bins, patches = plt.hist(thisPredProb[:,1], nBin, density=True, facecolor='r', alpha=0.75)
ax2.set_xlim(0.0,1.0)
ax2.set_ylim(0.0,maxY)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.set_title('Pred.Prob.: FWD')

ax3 = plt.subplot(3,3,3)
counts, bins, patches = plt.hist(thisPredProb[:,2], nBin, density=True, facecolor='r', alpha=0.75)
ax3.set_xlim(0.0,1.0)
ax3.set_ylim(0.0,maxY)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.set_title('Pred.Prob.: RWD')

thisIndex = y[y == 'FWD'].index
thisPredProb = y_predProb[thisIndex,:]
ax4 = plt.subplot(3,3,4)
counts, bins, patches = plt.hist(thisPredProb[:,0], nBin, density=True, facecolor='g', alpha=0.75)
ax4.set_xlim(0.0,1.0)
ax4.set_ylim(0.0,maxY)
plt.setp(ax4.get_xticklabels(), visible=False)
ax4.set_ylabel('Obs.: FWD')

ax5 = plt.subplot(3,3,5)
counts, bins, patches = plt.hist(thisPredProb[:,1], nBin, density=True, facecolor='g', alpha=0.75)
ax5.set_xlim(0.0,1.0)
ax5.set_ylim(0.0,maxY)
plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)

ax6 = plt.subplot(3,3,6)
counts, bins, patches = plt.hist(thisPredProb[:,2], nBin, density=True, facecolor='g', alpha=0.75)
ax6.set_xlim(0.0,1.0)
ax6.set_ylim(0.0,maxY)
plt.setp(ax6.get_xticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)

thisIndex = y[y == 'RWD'].index
thisPredProb = y_predProb[thisIndex,:]
ax7 = plt.subplot(3,3,7)
counts, bins, patches = plt.hist(thisPredProb[:,0], nBin, density=True, facecolor='b', alpha=0.75)
ax7.set_xlim(0.0,1.0)
ax7.set_ylim(0.0,maxY)
ax7.set_ylabel('Obs.: RWD')

ax8 = plt.subplot(3,3,8)
counts, bins, patches = plt.hist(thisPredProb[:,1], nBin, density=True, facecolor='b', alpha=0.75)
ax8.set_xlim(0.0,1.0)
ax8.set_ylim(0.0,maxY)
plt.setp(ax8.get_yticklabels(), visible=False)

ax9 = plt.subplot(3,3,9)
counts, bins, patches = plt.hist(thisPredProb[:,2], nBin, density=True, facecolor='b', alpha=0.75)
ax9.set_xlim(0.0,1.0)
ax9.set_ylim(0.0,maxY)
plt.setp(ax9.get_yticklabels(), visible=False)

plt.show()
