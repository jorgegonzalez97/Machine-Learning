import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as metrics

Y = numpy.array(['Event',
                 'Non-Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event'])

predProbY = numpy.array([0.9,0.5,0.3,0.7,0.3,0.8,0.4,0.2,1,0.5,0.3])

# Determine the predicted class of Y
predY = numpy.where(predProbY >= 0.5, 'Event', 'Non-Event')

# Calculate the Root Average Squared Error
Y_true = numpy.where(Y == 'Event', 1.0, 0.0)
ASE = numpy.mean(numpy.power(Y_true - predProbY, 2))
RASE = numpy.sqrt(ASE)

# Calculate the Root Mean Squared Error
RMSE = metrics.mean_squared_error(Y_true, predProbY)
RMSE = numpy.sqrt(RMSE)

# For binary y_true, y_score is supposed to be the score of the class with greater label.
AUC = metrics.roc_auc_score(Y_true, predProbY)
accuracy = metrics.accuracy_score(Y, predY)

print('                  Accuracy: {:.13f}' .format(accuracy))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
print('          Area Under Curve: {:.13f}' .format(AUC))
print('Root Average Squared Error: {:.13f}' .format(RASE))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE))

# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Event')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], fpr)
Sensitivity = numpy.append([0], tpr)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(dpi = 200)
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
plt.show()

# Draw the Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.NaN, thresholds)
plt.figure(dpi = 200)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()

print(thresholds)
print(tpr-fpr)

# Generate the coordinates for the Precision-Recall curve
Precision, Recall, thresholds = metrics.precision_recall_curve(Y, predProbY, pos_label = 'Event')

thresholds = numpy.append([0.2], thresholds)
Precision = numpy.append([6/11], Precision)
Recall = numpy.append([1.0], Recall)

# Draw the Precision-Recall curve
plt.figure(dpi = 200)
plt.plot(Recall, Precision, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [6/11, 6/11], color = 'red', linestyle = ':', label = 'No Skill')
plt.grid(True)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis("equal")
plt.show()

# Draw the F1 Score curve
F1 = 2.0 * (Precision * Recall) / (Precision + Recall)

a = F1[0:8]
plt.figure(dpi = 200)
plt.plot(thresholds, a, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.show()
