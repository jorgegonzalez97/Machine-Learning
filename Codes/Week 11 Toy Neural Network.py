import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.neural_network as nn
import sklearn.metrics as metrics

def pieceWise (x):
    if (x < 0.49):
        y = numpy.sqrt(x)
    else:
        y = 0.7 + 12.68742791 * (x - 0.49)**2
    return y

x = numpy.arange(0,201) / 200
y = numpy.array([pieceWise(xi) for xi in x])

# Plot the toy data
plt.figure(figsize=(10,6))
plt.plot(x, y, linewidth = 2, marker = '')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

xVar = pandas.DataFrame(x, columns = ['x'])

def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191030)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)

    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    
    # Plot the prediction
    plt.figure(figsize=(10,6))
    plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
    plt.plot(xVar, y_pred, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
    plt.legend(fontsize = 12, markerscale = 3)
    plt.show()
    
    return (Loss, RSquare)

result = pandas.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'RSquare'])

for i in numpy.arange(1,10):
    for j in numpy.arange(5,25,5):
        Loss, RSquare = Build_NN_Toy (nLayer = i, nHiddenNeuron = j)
        result = result.append(pandas.DataFrame([[i, j, Loss, RSquare]], 
                               columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'RSquare']))