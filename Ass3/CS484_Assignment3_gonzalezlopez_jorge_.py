##########################################
### CS484 Spring 2021
### Assignment 3
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
import statsmodels.api as stats
import scipy
from itertools import combinations


### Load the data
df_claim = pd.read_csv('claim_history.csv')

df_sample = pd.read_csv('sample_v10.csv')

### Preprocessing

#Function to get the optimal split out of a dataframe
def get_optimal_split(df,column,target,combinations):
	'''
		Parameter
		df  	 	 : Pandas dataframe.
		columns		 : Column of df for which the split is want to be carried out
		target 		 : Target column of df
		combinations : All possible combinations of the values of column

		Output
		best_split	 : Array containing the best combinations of values for the left and right split
		split_entropy: Entropy of the best split
		Gain		 : Gain in entropy of the best split 
	
	'''
	best_split = []
	Gain = 0
	split_entropy = 1

	#Get entropy of target value of all the samples in the dataset df
	probs_targ = df[target].value_counts(normalize=True)
	E_target = np.sum(-probs_targ * np.log2(probs_targ))    

	#For all possible combinations:
	for comb in combinations:

		#transform the combination into an array -> left split
		split_1 = []
		for values in comb:
			split_1.append(values)

		#Get the array for the values of the column no included in the combination -> right split
		split_2 = [ele for ele in df[column].unique() if ele not in split_1] 

		# Subgroup 1: dataset which values of column are contain in split1
		# Get probabilities of being Private / Commercial for subgroup1
		sub1 = df[df[column].isin(split_1)][target].value_counts(normalize=True)
		#Entropy of subgroup1
		E_1 = np.sum(-sub1 * np.log2(sub1))

		# Subgroup 2: dataset which values of column are contain in split2
		# Get probabilities of being Private / Commercial for subgroup2
		sub2 = df[df[column].isin(split_2)][target].value_counts(normalize=True)
		#Entropy of subgroup2
		E_2 = np.sum(-sub2 * np.log2(sub2))

		# Total number of samples in both subgroups
		T1 = len(df[df[column].isin(split_1)][target])
		T2 = len(df[df[column].isin(split_2)][target])

		# Calculate the split entropy (both entropies scale by the probability of ocurrence)
		E_s = E_1 * T1 / (T1+T2) + E_2 * T2 / (T1+T2)

		# Check and Store the split that gets the lower split entropy
		if E_s < split_entropy:
			Gain = E_target-E_s
			best_split = [split_1, split_2]
			split_entropy = E_s

	return best_split, split_entropy, Gain

def terminal_nodes(df, target):
	'''
		Parameter
		df  	 	: Pandas dataframe.
		target 		: Target column of df

		Output
		E			: Entropy of terminal node
		count 		: Total number of values for each category of the target
		probs_targ	: Probabilities of being every category of the target 

 	'''
	#Get the total number of Private / Commercial (target categories) samples in df
	count = df[target].value_counts()
	#Get the probability of Private / Commercial (target categories) in df
	probs_targ = df[target].value_counts(normalize=True)
	# Calculate the entropy
	E = np.sum(-probs_targ * np.log2(probs_targ))

	return E, count, probs_targ
    

def question2 (df):
	'''
		  Parameter
		  df_calim   : Pandas dataframe.

	 '''
	# Get relevant columns
	columns = df[['CAR_TYPE','OCCUPATION','EDUCATION', 'CAR_USE']]

	#Entropy of target value
	probs_targ = columns['CAR_USE'].value_counts(normalize=True)
	E_target = np.sum(-probs_targ * np.log2(probs_targ))

	print('\na) The entropy of the root node is: ', E_target)

	#All possible combinations of the values of the three relevant columns
	comb_occ = []
	for i in range(1,len(columns["OCCUPATION"].unique())):
	    comb_occ+=list(combinations(columns["OCCUPATION"].unique(),i))

	comb_type = []
	for i in range(1,len(columns["CAR_TYPE"].unique())):
	    comb_type+=list(combinations(columns["CAR_TYPE"].unique(),i))
	    
	comb_edu = [("Below High School",),("Below High School","High School",),("Below High School","High School","Bachelors",),("Below High School","High School","Bachelors","Masters",)]

	#Get the best split and its respective entropy out of the three features
	print('\nb) Left Split, Right Split, Entropy, Gain\n')
	print(get_optimal_split(columns,'CAR_TYPE','CAR_USE',comb_type))
	best_split, ent, gain = get_optimal_split(columns,'OCCUPATION','CAR_USE',comb_occ)
	print(best_split, ent, gain)
	print(get_optimal_split(columns,'EDUCATION','CAR_USE',comb_edu))

	print('\nC) OCCUPATION is the feature selected for splitting the first layer (higher gain)')

	#Repeat the same for the dataset that meets the split conditions
	
	print('\n Left Branch')

	# DATASET LEFT BRANCH: samples that meet the split condition
	Left_branch = columns[columns['OCCUPATION'].isin(best_split[0])]

	print(get_optimal_split(Left_branch,'CAR_TYPE','CAR_USE',comb_type))

	#All possible combinations (just for the left split values)
	comb_occ2 = []
	for i in range(1,len(best_split[0])):
	    comb_occ2+=list(combinations(best_split[0],i))

	print(get_optimal_split(Left_branch,'OCCUPATION','CAR_USE',comb_occ2))

	best_split_l, ent, gain = get_optimal_split(Left_branch,'EDUCATION','CAR_USE',comb_edu)

	print(best_split_l, ent, gain)

	#Repeat the same for the dataset that does not meet the split conditions

	print('\n Right Branch')

	# DATASET RIGHT BRANCH: samples that do not meet the split condition
	right_branch = columns[columns['OCCUPATION'].isin(best_split[1])]

	best_split_r, ent, gain = get_optimal_split(right_branch,'CAR_TYPE','CAR_USE',comb_type)

	print(best_split_r, ent, gain)

	#All possible combinations (just for the right split values)
	comb_occ2 = []
	for i in range(1,len(best_split[1])):
	    comb_occ2+=list(combinations(best_split[1],i))

	print(get_optimal_split(right_branch,'CAR_TYPE','CAR_USE',comb_occ2))

	print(get_optimal_split(right_branch,'CAR_TYPE','CAR_USE',comb_edu))

	print('\nd) EDUCATION  and CAR_TYPE are the features selected for splitting the second layer left and right branches respectively (higher gains)')

	print('\nTerminal Nodes')

	# LAST DATASETS: 
	# samples that meet the split condition
	left_left = Left_branch[Left_branch['EDUCATION'].isin(best_split_l[0])]
	# samples that do not meet the split condition
	left_right = Left_branch[Left_branch['EDUCATION'].isin(best_split_l[1])]

	# samples that meet the split condition
	right_left = right_branch[right_branch['CAR_TYPE'].isin(best_split_r[0])]
	# samples that do not meet the split condition
	right_right = right_branch[right_branch['CAR_TYPE'].isin(best_split_r[1])]

	E, count, prob = terminal_nodes(left_left, 'CAR_USE')
	print('\n1. Terminal node: Entropy: ', E, '. Private: ' ,count[0], '. Commercial: ' ,count[1],'. Total: ' ,count[0]+count[1], 'Probabilities of Private: ', prob[0],'Probabilities of Commercial: ', prob[1] )
	E, count, prob = terminal_nodes(left_right, 'CAR_USE')
	print('\n2. Terminal node: Entropy: ', E, '. Private: ' ,count[0], '. Commercial: ' ,count[1],'. Total: ' ,count[0]+count[1], 'Probabilities of Private: ', prob[0],'Probabilities of Commercial: ', prob[1] )
	E, count, prob = terminal_nodes(right_left, 'CAR_USE')
	print('\n3. Terminal node: Entropy: ', E, '. Private: ' ,count[0], '. Commercial: ' ,count[1],'. Total: ' ,count[0]+count[1], 'Probabilities of Private: ', prob[0],'Probabilities of Commercial: ', prob[1] )
	E, count, prob = terminal_nodes(right_right, 'CAR_USE')
	print('\n4. Terminal node: Entropy: ', E, '. Private: ' ,count[0], '. Commercial: ' ,count[1],'. Total: ' ,count[0]+count[1], 'Probabilities of Private: ', prob[0],'Probabilities of Commercial: ', prob[1] )





def question3 (df):
	'''
	Parameter
	df_cars: Pandas dataframe

	'''

	# Make sure the target is a category type of value
	df['y'] = df['y'].astype('category')

	print('a) The frequency table of the categorical target field is: ')
	print(df['y'].value_counts())

	#Initial Model of Backward Selection

	y = df['y']
	X = df.drop(['y'], axis=1)

	columns = X.columns

	X = stats.add_constant(X, prepend=True)

	DF1 = np.linalg.matrix_rank(X) * (len(y.value_counts()) - 1)

	logit = stats.MNLogit(y, X)
	thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
	thisParameter = thisFit.params
	LLK1 = logit.loglike(thisParameter.values)

	print(thisFit.summary())
	print("Model Log-Likelihood Value =", LLK1)
	print("Number of Free Parameters =", DF1)

	#Check the relevance of the features: backward method
	# The feature with the worse significance is eliminated iteratively
	# until there are only features with a significance lower than 0.05

	removed_columns = []
	best_model = []
	i = 1

	while True:
		y = df['y']
		X = df.drop(['y'], axis=1)
		X = X.drop(removed_columns, axis = 1)

		columns = X.columns

		X = stats.add_constant(X, prepend=True)

		DF1 = np.linalg.matrix_rank(X) * (len(y.value_counts()) - 1)

		logit = stats.MNLogit(y, X)
		thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
		thisParameter = thisFit.params
		LLK1 = logit.loglike(thisParameter.values)

		worse_col = []
		bigger_pvalue = 0
		LLK0_saved = 0
		DF0_saved = 0
		Dev_saved = 0
		DF_saved = 0
		AIC_saved = 0
		BIC_saved = 0
    
    
		for col in columns:

			X2 = X.drop(col, axis = 1)
			X2 = stats.add_constant(X2, prepend=True)

			DF0 = np.linalg.matrix_rank(X2) * (len(y.value_counts()) - 1)

			logit = stats.MNLogit(y, X2)
			thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
			thisParameter = thisFit.params
			LLK0 = logit.loglike(thisParameter.values)

			Deviance = 2 * (LLK1 - LLK0)
			DF = DF1 - DF0
			pValue = scipy.stats.chi2.sf(Deviance, DF)

			MDF = (thisFit.J - 1) * thisFit.K
			LLK = thisFit.llf

			NSample = len(y)
			AIC = 2.0 * MDF - 2.0 * LLK
			BIC = MDF * np.log(NSample) - 2.0 * LLK

			if pValue > bigger_pvalue:
				worse_col = col
				bigger_pvalue = pValue
				LLK0_saved = LLK0
				DF0_saved = DF0
				Dev_saved = Deviance
				DF_saved = DF
				AIC_saved = AIC
				BIC_saved = BIC

    
		if bigger_pvalue < 0.05:
			best_model = X2.columns
			print('\n Best model --> Removed feature: ', removed_columns, ' if we try to remove another feature: ', worse_col)
			print("Model Log-Likelihood Value =", LLK0_saved)
			print("Number of Free Parameters =", DF0_saved)
			print("Deviance (Statistic, DF, Significance)", Dev_saved, DF_saved, bigger_pvalue)
			print("Akaike Information Criterion =", AIC_saved)
			print("Bayesian Information Criterion =", BIC_saved)
			print('\n')
			break
    
		removed_columns.append(worse_col)
		print('\nModel', i, ' --> Removed feature: ', removed_columns)
		print("Model Log-Likelihood Value =", LLK0_saved)
		print("Number of Free Parameters =", DF0_saved)
		print("Deviance (Statistic, DF, Significance)", Dev_saved, DF_saved, bigger_pvalue)
		print("Akaike Information Criterion =", AIC_saved)
		print("Bayesian Information Criterion =", BIC_saved)    
		print('\n')
		i += 1


	#Final model with backward selection methos
	y = df['y']
	X = df.drop(['y'], axis=1)
	X = X.drop(removed_columns, axis = 1)
	X = stats.add_constant(X, prepend=True)

	DF1 = np.linalg.matrix_rank(X) * (len(y.value_counts()) - 1)

	logit = stats.MNLogit(y, X)
	thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
	thisParameter = thisFit.params
	LLK1 = logit.loglike(thisParameter.values)

	print(thisFit.summary())
	print("Model Log-Likelihood Value =", LLK1)
	print("Number of Free Parameters =", DF1)




print('\n-- Question 2 --\n')

question2(df_claim)

print('\n-- Question 3 --\n')

question3(df_sample)