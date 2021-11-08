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


### Load the data
df_pur = pd.read_csv('Purchase_Likelihood.csv')

df_spi = pd.read_csv('SpiralWithCluster.csv')

#Function to compute the Cramer's V statistics with a crosstabulation table.
def cramerV(ctab):
	n_ip = np.sum(np.array(ctab), axis = 1, keepdims = True)
	n_pj = np.sum(np.array(ctab), axis = 0, keepdims = True)
	n_pp = np.sum(np.array(ctab))

	E_ij = np.dot(n_ip, n_pj)/n_pp

	P_chi = np.sum(np.divide(np.square(np.array(ctab) - E_ij), E_ij))

	return np.sqrt(P_chi / (n_pp * (np.min(np.array(ctab).shape)-1)))
  

def question1 (df):
	'''
		  Parameter
		  df  : Pandas dataframe.

	 '''
	# Get frequency counts and class probs of the target variable. 
	print('\n a) \n')

	print(df['insurance'].value_counts())
	print(df['insurance'].value_counts(normalize = True))

	# Get crosstabulation tables of the target variable with every feature independently

	print('\n b) \n')
	#print(df.groupby(['group_size','insurance']).size())
	g_t = pd.crosstab(df['group_size'], df['insurance'])
	print(g_t)

	print('\n c) \n')
	#print(df.groupby(['homeowner','insurance']).size())
	h_t = pd.crosstab(df['homeowner'], df['insurance']) 
	print(h_t)

	print('\n d) \n')
	#print(df.groupby(['homeowner','insurance']).size())
	m_t = pd.crosstab(df['married_couple'], df['insurance'])
	print(m_t)

	#Get the Cramer's V statistics of each crosstab table
	cram_g = cramerV(g_t)
	cram_h = cramerV(h_t)
	cram_m = cramerV(m_t)

	print('\n e) \n')
	print(" Feature    Cramer's V\n")
	print("group_size ", cram_g)
	print("homeowner ", cram_h)
	print("married_couple ", cram_m)	

	# Train a naive_bayes model
	yTrain = df['insurance']

	xTrain = df[['group_size', 'homeowner', 'married_couple']]

	_objNB = naive_bayes.CategoricalNB(alpha = 1.e-10)
	thisModel = _objNB.fit(xTrain, yTrain)

	#All the possible combinations of the features' values
	list_posib = []
	for i in df['group_size'].unique():
	    for j in df['homeowner'].unique():
	        for t in df['married_couple'].unique():
	            list_posib.append([i,j,t])
	list_posib.sort()

	# Create dataframe with all the possible combinations of the features' values
	xTest = pd.DataFrame(list_posib, columns = ['group_size', 'homeowner', 'married_couple'])

	# Score the xTest and append the predicted probabilities to the xTest
	yTest_predProb = pd.DataFrame(_objNB.predict_proba(xTest), columns = ['P_ins0', 'P_ins1','P_ins2'])
	yTest_score = pd.concat([xTest, yTest_predProb], axis = 1)

	print('\n f) \n')
	print(yTest_score)

	#Compute values of odd requested:  Prob(insurance = 1) / Prob(insurance = 2)
	odds = np.array(yTest_score['P_ins1'] / yTest_score['P_ins2'])

	#Get combiantion with maximum value of the odds
	comb = yTest_score[['group_size', 'homeowner', 'married_couple']].iloc[np.argmax(odds)]

	print('\n g) \n')
	print(comb)

	print('\n')
	print('Maximum odds value: ', np.max(odds))



def question2 (df):
	'''
	Parameter
	df: Pandas dataframe

	'''

	xTrain = df[['x','y']]
	yTrain = df['SpectralCluster']

	svm_Model = svm.SVC(kernel = 'linear', decision_function_shape='ovr', random_state = 20210325, max_iter = -1)
	thisFit = svm_Model.fit(xTrain, yTrain)

	y_predictClass = thisFit.predict(xTrain)

	# get the separating hyperplane
	w = thisFit.coef_[0]
	a = -w[0] / w[1]
	xx = np.linspace(-4, 4)
	yy = a * xx - (thisFit.intercept_[0]) / w[1]

	print('\n a) \n')
	print('The intercept w0 is: ', thisFit.intercept_)
	print('The coeficients are: ', thisFit.coef_)
	print('The equation is: ', np.round(thisFit.intercept_[0],decimals=7), ' + ', np.round(w[0],decimals=7), 'x + ', np.round(w[1],decimals=7), 'y = 0')

	print('\n b) \n')
	print('Misclassification Rate = ', 1 - metrics.accuracy_score(yTrain, y_predictClass))


	# Plot scatter plot of y against x with the hyperplane
	plt.figure(figsize=(5,5))

	plt.scatter(df[df['SpectralCluster']==0]['x'],df[df['SpectralCluster']==0]['y'], c = 'r')
	plt.scatter(df[df['SpectralCluster']==1]['x'],df[df['SpectralCluster']==1]['y'], c = 'b')

	plt.plot(xx, yy, 'g:')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Scatter plot of y-coordinate against x-coordinate')
	plt.legend([ 'Hyperplane','SpectralCluster = 0', 'SpectralCluster = 1'])
	plt.grid(True)
	plt.show()

	# Compute the r and theta values with x and y coordinates
	df['r'] = np.sqrt(df['x']**2 +  df['y']**2)
	df['theta'] = np.arctan2(df['y'], df['x'])

	#Shift the negative values of the arctan
	df['theta'] = df['theta'].apply(lambda x: 2.0*np.pi + x if x < 0.0 else x)

	# Plot scatter plot of theta against r with different colors for the SpectralCluster values (0,1)
	plt.figure(figsize=(5,5))
	plt.scatter(df[df['SpectralCluster']==0]['r'],df[df['SpectralCluster']==0]['theta'], c = 'r')
	plt.scatter(df[df['SpectralCluster']==1]['r'],df[df['SpectralCluster']==1]['theta'], c = 'b')

	plt.xlabel('r')
	plt.ylabel('Theta')
	plt.title('Scatter plot of theta against r')
	plt.legend(['SpectralCluster = 0', 'SpectralCluster = 1'])
	plt.grid(True)
	plt.show()

	# Create group values (0,1,2,3) to split the data  with polar coordinates
	df['Group'] = df['SpectralCluster']

	for i in range(len(df['Group'])):
		r = df['r'].iloc[i]
		theta = df['theta'].iloc[i]
		clas = df['SpectralCluster'].iloc[i]
		if r < 2 and theta > 6:
			df.at[i, 'Group'] = 0
		elif r < 3 and theta > 3 and clas == 1:
			df.at[i, 'Group'] = 1
		elif clas == 0:
			df.at[i, 'Group'] = 2
		else:
			df.at[i, 'Group'] = 3

	# Plot scatter plot of theta against r with different colors for the Group values (0,1,2,3) 
	plt.figure(figsize=(5,5))
	plt.scatter(df[df['Group']==0]['r'],df[df['Group']==0]['theta'], c = 'r')
	plt.scatter(df[df['Group']==1]['r'],df[df['Group']==1]['theta'], c = 'b')
	plt.scatter(df[df['Group']==2]['r'],df[df['Group']==2]['theta'], c = 'g')
	plt.scatter(df[df['Group']==3]['r'],df[df['Group']==3]['theta'], c = 'k')


	plt.xlabel('r')
	plt.ylabel('Theta')
	plt.title('Scatter plot of theta against r')
	plt.legend(['Group = 0', 'Group = 1','Group = 2','Group = 3'])
	plt.grid(True)
	plt.show()

	# Train SVM models for each pair of groups (0,1) (1,2) (2,3) and stor hyperplanes      
	print('\n f) \n')
	hyperplanes = []
	for i in range(3):
		print('SVM', i)
		xTrain = df[['r','theta']].loc[(df['Group']==i) | (df['Group']== i+1)]
		yTrain = df['SpectralCluster'].loc[(df['Group']==i) | (df['Group']== i+1)]

		svm_Model = svm.SVC(kernel = 'linear', decision_function_shape='ovr', random_state = 20210325, max_iter = -1)
		thisFit = svm_Model.fit(xTrain, yTrain)

		y_predictClass = thisFit.predict(xTrain)

		# get the separating hyperplane
		w = thisFit.coef_[0]
		a = -w[0] / w[1]
		xx = np.linspace(1, 4.5)
		yy = a * xx - (thisFit.intercept_[0]) / w[1]

		hyperplanes.append([xx,yy])

		#Print the three hyperplanes equattions
		print('The intercept w0 is: ', thisFit.intercept_)
		print('The coeficients are: ', thisFit.coef_)
		print('The equation is: ', np.round(thisFit.intercept_[0],decimals=7), ' + ', np.round(w[0],decimals=7), 'r + ', np.round(w[1],decimals=7), 'theta = 0')

	# Plot scatter plot of theta against r with different colors for the Group values (0,1,2,3) and the three hyperplanes
	plt.figure(figsize=(5,5))
	plt.scatter(df[df['Group']==0]['r'],df[df['Group']==0]['theta'], c = 'r')
	plt.scatter(df[df['Group']==1]['r'],df[df['Group']==1]['theta'], c = 'b')
	plt.scatter(df[df['Group']==2]['r'],df[df['Group']==2]['theta'], c = 'g')
	plt.scatter(df[df['Group']==3]['r'],df[df['Group']==3]['theta'], c = 'k')

	plt.plot(hyperplanes[0][0], hyperplanes[0][1], 'r:')
	plt.plot(hyperplanes[1][0], hyperplanes[1][1], 'b:')
	plt.plot(hyperplanes[2][0], hyperplanes[2][1], 'k:')

	plt.xlabel('r')
	plt.ylabel('Theta')
	plt.title('Scatter plot of theta against r')
	plt.legend(['hyperplane 1','hyperplane 2','hyperplane 3', 'Group = 0', 'Group = 1','Group = 2','Group = 3'])
	plt.grid(True)
	plt.show()

	# Plot scatter plot of x against y with different colors for the SpectralCluster values (0,1) and hypercureves (previous hyerplanes back to Cartesian coordinates)
	plt.figure(figsize=(5,5))

	plt.scatter(df[df['SpectralCluster']==0]['x'],df[df['SpectralCluster']==0]['y'], c = 'r')
	plt.scatter(df[df['SpectralCluster']==1]['x'],df[df['SpectralCluster']==1]['y'], c = 'b')

	h0_xx = hyperplanes[0][0] * np.cos(hyperplanes[0][1])
	h0_yy = hyperplanes[0][0] * np.sin(hyperplanes[0][1])

	h1_xx = hyperplanes[1][0] * np.cos(hyperplanes[1][1])
	h1_yy = hyperplanes[1][0] * np.sin(hyperplanes[1][1])

	h2_xx = hyperplanes[2][0] * np.cos(hyperplanes[2][1])
	h2_yy = hyperplanes[2][0] * np.sin(hyperplanes[2][1])

	plt.plot(h0_xx, h0_yy, 'r:')
	plt.plot(h1_xx, h1_yy, 'b:')
	plt.plot(h2_xx, h2_yy, 'k:')

	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Scatter plot of y-coordinate against x-coordinate')
	plt.legend(['hypercurve 1','hypercurve 2','hypercurve 3', 'SpectralCluster = 0', 'SpectralCluster = 1'])
	plt.show()




print('\n-- Question 1 --\n')

question1(df_pur)

print('\n-- Question 2 --\n')

question2(df_spi)