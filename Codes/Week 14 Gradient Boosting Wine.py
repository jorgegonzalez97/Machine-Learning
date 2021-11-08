import pandas
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import statsmodels.api as stats

WineQuality = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\WineQuality.csv', delimiter=',')

WineQuality = WineQuality.dropna()

WQ_size = WineQuality.groupby('quality_grp').size()

X_name = ['fixed_acidity', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide',
          'total_sulfur_dioxide', 'pH', 'sulphates']

# Build a logistic regression
y = WineQuality['quality_grp'].astype('category')
y_category = y.cat.categories

X = WineQuality[X_name]
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

y_confusion = metrics.confusion_matrix(y, y_predictClass)
print("Confusion Matrix (Row is Data, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predictClass)
print("Accuracy Score = ", y_accuracy)

# Perform the Gradient Boosting
X = WineQuality[X_name]
gbm = ensemble.GradientBoostingClassifier(loss='deviance', criterion='mse', n_estimators = 1000,
                                          max_leaf_nodes = 10, verbose=1)
fit_gbm = gbm.fit(X, y)
predY_gbm = gbm.predict(X)
Accuracy_gbm = gbm.score(X, y)
print('Gradient Boosting Model:', 'Accuracy = ', Accuracy_gbm)

# Plot the data
y_confusion = metrics.confusion_matrix(y, predY_gbm)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)
