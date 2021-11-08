import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.svm as svm

trainData = pandas.read_excel('C:\\IIT\\Machine Learning\\Data\\ThreeSegment.xlsx',
                              sheet_name = 'ThreeSegment', usecols = ['Group', 'X', 'Y'])

# Scatterplot that uses prior information of the grouping variable
carray = ['red', 'green', 'blue']
plt.figure(figsize=(10,10))
for i in range(3):
    subData = trainData[trainData['Group'] == (i+1)]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = (i+1), s = 25)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Build Support Vector Machine classifier
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
for j in range(3):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green', 'blue']
plt.figure(figsize=(10,10))
for i in range(3):
    subData = trainData[trainData['_PredictedClass_'] == (i+1)]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = (i+1), s = 25)
plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Support Vector Machines on Three Segments')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-7,7)
plt.ylim(-7,7)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Convert to the polar coordinates
trainData['radius'] = numpy.sqrt(trainData['X']**2 + trainData['Y']**2)
trainData['theta'] = numpy.arctan2(trainData['Y'], trainData['X'])

def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

trainData['theta'] = trainData['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = trainData[['radius','theta']]
yTrain = trainData['Group']

print(xTrain.isnull().sum())

# Scatterplot that uses prior information of the grouping variable
carray = ['red', 'green', 'blue']
plt.figure(figsize=(10,10))
for i in range(3):
    subData = trainData[trainData['Group'] == (i+1)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i+1), s = 25)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

xTrain = trainData[['radius','theta']]
yTrain = trainData['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(0, 6)
yy = numpy.zeros((len(xx),3))
for j in range(3):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green', 'blue']
plt.figure(figsize=(10,10))
for i in range(3):
    subData = trainData[trainData['_PredictedClass_'] == (i+1)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i+1), s = 25)
plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Support Vector Machines on Three Segments')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.ylim(-0.5, 6.5)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


h0_xx = xx * numpy.cos(yy[:,0])
h0_yy = xx * numpy.sin(yy[:,0])

h1_xx = xx * numpy.cos(yy[:,1])
h1_yy = xx * numpy.sin(yy[:,1])

h2_xx = xx * numpy.cos(yy[:,2])
h2_yy = xx * numpy.sin(yy[:,2])

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green', 'blue']
plt.figure(figsize=(10,10))
for i in range(3):
    subData = trainData[trainData['_PredictedClass_'] == (i+1)]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = (i+1), s = 25)
plt.plot(h0_xx, h0_yy, color = 'black', linestyle = '-')
plt.plot(h1_xx, h1_yy, color = 'black', linestyle = '-')
plt.plot(h2_xx, h2_yy, color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Support Vector Machines on Three Segments')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
