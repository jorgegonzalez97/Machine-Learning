##########################################
### CS484 Spring 2021
### Assignment 4
### Student ID: A20474413
### Jorge Gonzalez Lopez


### 1/24/2020 -- Initial commit
### 1/24/2020 -- Finalizing question 1
### 1/25/2020 -- Finished assignment
##########################################



### Load the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn import naive_bayes, svm, metrics
from sklearn.linear_model import LogisticRegression
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import statsmodels.api as stats
import scipy
import random



### Load the data
df = pd.read_csv('WineQuality_Train.csv')
df_test = pd.read_csv('WineQuality_Test.csv')

input_features = ['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']
target = 'quality_grp'


def SWEEPOperator (pDim, inputM, tol):
	# pDim: dimension of matrix inputM, integer greater than one
	# inputM: a square and symmetric matrix, numpy array
	# tol: singularity tolerance, positive real

	aliasParam = []
	nonAliasParam = []

	A = np.copy(inputM)
	diagA = np.diagonal(inputM)

	for k in range(pDim):
		Akk = A[k,k]
		if (Akk >= (tol * diagA[k])):
			nonAliasParam.append(k)
			ANext = A - np.outer(A[:, k], A[k, :]) / Akk
			ANext[:, k] = A[:, k] / Akk
			ANext[k, :] = ANext[:, k]
			ANext[k, k] = -1.0 / Akk
		else:
			aliasParam.append(k)
			ANext[:, k] = 0.0 * A[:, k]
			ANext[k, :] = ANext[:, k]
		A = ANext
	return (A, aliasParam, nonAliasParam)


def build_mnlogit (fullX, y):

	# Find the non-redundant columns in the design matrix fullX
	nFullParam = fullX.shape[1]
	XtX = np.transpose(fullX).dot(fullX)
	invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-7)

	# Build a multinomial logistic model
	X = fullX.iloc[:, list(nonAliasParam)]
	logit = stats.MNLogit(y, X)
	thisFit = logit.fit(method = 'newton', maxiter = 1000, gtol = 1e-6, full_output = True, disp = True)
	thisParameter = thisFit.params
	thisLLK = logit.loglike(thisParameter.values)

	# The number of free parameters
	nYCat = thisFit.J
	thisDF = len(nonAliasParam) * (nYCat - 1)

	# Return model statistics
	return (thisLLK, thisDF, thisParameter, thisFit)

def sample_wr (n):
	outData = []
	for i in range(n):
		j = int(random.random() * n)
		outData.append(j)
	return outData

def question1 (df, df_test):
	'''
		  Parameter
		  df  : Pandas dataframe.

	 '''

	X = df[input_features]
	y = df[target].astype('category')

	X_test = df_test[input_features]
	y_test = df_test[target].astype('category')

	crit = 'entropy'
	max_d = 5
	rnd_st = 20210415
	max_ite = 50
 
	print('\n a) b) c) \n')

	w_train = np.full(X.shape[0], 1.0)
	accuracy = []
	accuracy_test = []
	ensemblePredProb = np.zeros((X.shape[0], 2))

	ensemblePredProb_test = np.zeros((X_test.shape[0], 2))


	for iter in range(max_ite):
		classTree = tree.DecisionTreeClassifier(criterion=crit, max_depth=max_d, random_state=rnd_st)
		treeFit = classTree.fit(X, y, w_train)

		treePredProb = classTree.predict_proba(X)
		y_pred = np.where(treePredProb[:,1] >= 0.2, 1, 0)
		accuracy.append(np.sum(np.where(y_pred == y, w_train, 0)) / np.sum(w_train))
		ensemblePredProb += accuracy[iter] * treePredProb

		treePredProb_test = classTree.predict_proba(X_test)
		y_pred_test = np.where(treePredProb_test[:,1] >= 0.2, 1, 0)
		accuracy_test.append(np.mean(np.where(y_pred_test == y_test, 1, 0)))
		ensemblePredProb_test += accuracy_test[iter] * treePredProb_test


		if (abs(accuracy[iter]) >= 0.9999999):
			print('\n Iteration ', iter ,': Max accuracy achieved')
			break

		# Update the weights
		eventError = np.where(y == 1, (1 - treePredProb[:,1]), treePredProb[:,1])
		predClass = np.where(treePredProb[:,1] >= 0.2, 1, 0)
		w_train = np.where(predClass != y, 2+np.abs(eventError), np.abs(eventError))

		if iter in [0,1]:
			print('\n Misclassification rate for iteration ', iter, ' :')
			print('\n ', 1-accuracy[iter])
	    

	# Calculate the final predicted probabilities
	ensemblePredProb /= np.sum(accuracy)
	ensemblePredProb_test /= np.sum(accuracy_test)

	#final_pred = np.where(ensemblePredProb[:,1] >= 0.2, 1, 0)
	print('\n Final Misclassification rate:')
	print('\n ', 1 - accuracy[-1])


	print('\n d) \n')
	final_pred_test = np.where(ensemblePredProb_test[:,1] >= 0.2, 1, 0)
	AUC = metrics.roc_auc_score(y_test, final_pred_test)

	print('AUC = ', AUC)


	df_test['Predictions_1'] = ensemblePredProb_test[:,1]


	print('\n e) \n')
	print('Figure')

	df_test.boxplot(column='Predictions_1', by=target, vert = False, figsize=(6,4))
	plt.title("Boxplot of Predicitons_1")
	plt.suptitle(" ")
	plt.xlabel("Predictions")
	plt.ylabel("quality_grp")
	plt.grid(axis="y")
	plt.show()



def question2 (df, df_test):
	'''
	Parameter
	df: Pandas dataframe

	'''

	X = df[input_features]
	y = df[target].astype('category')

	devianceTable = pd.DataFrame()

	u = pd.DataFrame()

	# Step 0: Intercept only model
	u = y.isnull()
	designX = pd.DataFrame(u.where(u, 1)).rename(columns = {target: "const"})
	LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)
	devianceTable = devianceTable.append([[0, 'Intercept', DF0, LLK0, None, None, None]])

	for i in input_features:
		# Step 1: Intercept + one_feature
		designX = df[i]
		designX = stats.add_constant(designX, prepend = True)
		LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
		testDev = 2.0 * (LLK1 - LLK0)
		testDF = DF1 - DF0
		testPValue = scipy.stats.chi2.sf(testDev, testDF)
		devianceTable = devianceTable.append([[1, 'Intercept + ' + i, DF1, LLK1, testDev, testDF, testPValue]])

	features2 = ['citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']
	for i in features2:
		# Step 2: Intercept + two_feature
		designX = df[['alcohol',i]]
		designX = stats.add_constant(designX, prepend = True)
		LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
		testDev = 2.0 * (LLK1 - LLK0)
		testDF = DF1 - DF0
		testPValue = scipy.stats.chi2.sf(testDev, testDF)
		devianceTable = devianceTable.append([[2, 'Intercept + alcohol + ' + i, DF1, LLK1, testDev, testDF, testPValue]])

	features2 = ['citric_acid', 'residual_sugar', 'sulphates']
	for i in features2:
		# Step 3: Intercept + 3_feature
		designX = df[['alcohol','free_sulfur_dioxide',i]]
		designX = stats.add_constant(designX, prepend = True)
		LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
		testDev = 2.0 * (LLK1 - LLK0)
		testDF = DF1 - DF0
		testPValue = scipy.stats.chi2.sf(testDev, testDF)
		devianceTable = devianceTable.append([[3, 'Intercept + alcohol + free_sulfur_dioxide + ' + i, DF1, LLK1, testDev, testDF, testPValue]])

	features2 = ['citric_acid', 'residual_sugar']
	for i in features2:
		# Step 1: Intercept + one_feature
		designX = df[['alcohol','free_sulfur_dioxide', 'sulphates', i]]
		designX = stats.add_constant(designX, prepend = True)
		LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
		testDev = 2.0 * (LLK1 - LLK0)
		testDF = DF1 - DF0
		testPValue = scipy.stats.chi2.sf(testDev, testDF)
		devianceTable = devianceTable.append([[5, i + 'Intercept + alcohol + free_sulfur_dioxide + sulphates', DF1, LLK1, testDev, testDF, testPValue]])


	features2 = ['citric_acid']
	for i in features2:
		# Step 1: Intercept + one_feature
		designX = df[['alcohol','free_sulfur_dioxide', 'sulphates', 'residual_sugar' , i]]
		designX = stats.add_constant(designX, prepend = True)
		LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
		testDev = 2.0 * (LLK1 - LLK0)
		testDF = DF1 - DF0
		testPValue = scipy.stats.chi2.sf(testDev, testDF)
		devianceTable = devianceTable.append([[6, i + 'Intercept + alcohol + free_sulfur_dioxide + sulphates + residual_sugar', DF1, LLK1, testDev, testDF, testPValue]])

	print(devianceTable)

	X_test = df_test[input_features]
	X_test = stats.add_constant(X_test, prepend = True)
	y_test = df_test[target].astype('category')

	X = df[input_features]
	X = stats.add_constant(X, prepend = True)

	logit = stats.MNLogit(y, X)
	thisFit = logit.fit(method = 'newton', maxiter = 1000, gtol = 1e-6, full_output = True, disp = True)
	pred = thisFit.predict(X_test)

	AUC = metrics.roc_auc_score(y_test, pd.to_numeric(pred.idxmax(axis=1)))

	print('\n AUC = ', AUC)

	X_train = df[input_features]
	X_train = stats.add_constant(X_train, prepend = True)
	y_train = df[target].astype('category')

	auc_total =[]
	
	random.seed(20210415)
	for i in range(10000):
		sample = sample_wr(len(X_train))

		X_sample = X.iloc[sample]
		y_sample = y.iloc[sample]

		logit = stats.MNLogit(y_sample, X_sample)
		thisFit = logit.fit(maxiter = 100)

		pred = thisFit.predict(X_test)
		pred = pd.to_numeric(pred.idxmax(axis=1))
		AUC = metrics.roc_auc_score(y_test, pred)

		auc_total.append(AUC)

    
	binwidth = 0.001
	plt.hist(auc_total,bins=np.arange(np.min(auc_total), np.max(auc_total) + binwidth, binwidth))
	plt.xlabel('AUC')
	plt.ylabel('Frequency')
	plt.show()

	print('\n2.5th percentile: ', np.percentile(auc_total, 2.5))
	print('97.5th percentile: ', np.percentile(auc_total, 97.5))

print('\n-- Question 1 --\n')

question1(df, df_test)

print('\n-- Question 2 --\n')

question2(df, df_test)