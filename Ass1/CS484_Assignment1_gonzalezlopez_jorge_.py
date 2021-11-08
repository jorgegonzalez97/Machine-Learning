##########################################
### CS484 Spring 2021
### Assignment 1
### Student ID: A20474413
### Jorge Gonzalez Lopez


### 1/24/2020 -- Initial commit
### 1/24/2020 -- Finalizing question 1
### 1/25/2020 -- Finished assignment
##########################################



### Load the necessary libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.linalg as sl
from sklearn.neighbors import KNeighborsClassifier


### Load the data
data = pd.read_csv('NormalSample.csv', delimiter=',')

df = pd.read_csv('fraud.csv', delimiter=',')

d_a = pd.read_csv('airplanes.csv', delimiter = ';')
d_a = d_a.fillna('___')
### Preprocessing


### Define a function
def question1 (data):
     '''
          Parameter
          data     : Pandas dataframe.

          Output
          desc    : description of the column x from data.
          h_iz    : Bin width recommended by the Izenman
          C_min	  : minimum C  by the Shimazaki and Shinomoto.
          D[idx]  : bin width that minimizes C by Shimazaki and SHinomoto.
          binMId  : midpoints for Shimazaki and Shinomoto with recommended bin width.
     '''
     desc = data['x'].describe()
     print('a) Use the Pandas describe on the field x in the NormalSample.csv')
     print(desc)

     # Calculate Izenman bin width with the formula
     print('\nb) Bin width recommended by the Izenman (1991) method')
     h_iz = 2 * (desc['75%'] - desc['25%']) * (desc['count']**(-1/3))
     h_iz = np.round(h_iz,decimals=1)
     print(h_iz)

     # Use the vector of bin widths D to estimate the best one with Shinomoto
     result = pd.DataFrame()
     D = [0.1, 0.2, 0.5, 1, 2, 5]

     for d in D:
         nBin, middleY, lowY, CDelta = calcCD(data['x'],d)
         highY = lowY + nBin * d
         result = result.append([[d, CDelta, lowY, middleY, highY, nBin]], ignore_index = True)
 
     # Get the lower C and its index of all bin widths
     C_min = np.min(result[1])
     idx  = np.where(result[1]==C_min)[0][0]    
     print('c) Bin width recommended by the Shimazaki and Shinomoto (2007) method:')
     print('The minimum C is: '+ str(np.round(C_min,decimals=2)) + ' and the bin width that minimizes C is: '+ str(D[idx]))

     # Plot histogram with best bin width
     nBin, middleY, lowY, CDelta = calcCD(data['x'],D[idx])
     highY = lowY + nBin * D[idx]
     binMid = lowY + 0.5 * D[idx] + np.arange(nBin) * D[idx]
     plt.hist(data['x'], bins = binMid, align='mid')
     plt.title('Delta = ' + str(D[idx]))
     plt.ylabel('Number of Observations')
     plt.grid(axis = 'y')

     print('d) The midpoints are: ' + str(binMid))
     plt.show()

     return(desc, h_iz, C_min, D[idx], binMid)



def calcCD (Y, delta):
	'''
		  Fucntion to calculate the value of C for a bin width with
		  the Shimazaki and Shinomoto method

		  Parameter
		  Y     : Pandas dataframe.
		  delta : bid width.

		  Output
		  m    : number of bins
		  middleY    : middle value of Y 
		  lowY	  : lower value of Y
		  CDelta  : value of C
	 '''
	maxY = np.max(Y)
	minY = np.min(Y)
	meanY = np.mean(Y)

   # Round the mean to integral multiples of delta
	middleY = delta * np.round(meanY / delta)

   # Determine the number of bins on both sides of the rounded mean
	nBinRight = np.ceil((maxY - middleY) / delta)
	nBinLeft = np.ceil((middleY - minY) / delta)
	lowY = middleY - nBinLeft * delta

   # Assign observations to bins starting from 0
	m = nBinLeft + nBinRight
	BIN_INDEX = 0;
	boundaryY = lowY
	for iBin in np.arange(m):
		boundaryY = boundaryY + delta
		BIN_INDEX = np.where(Y > boundaryY, iBin+1, BIN_INDEX)

   # Count the number of observations in each bins
	uBin, binFreq = np.unique(BIN_INDEX, return_counts = True)

   # Calculate the average frequency
	meanBinFreq = np.sum(binFreq) / m
	ssDevBinFreq = np.sum((binFreq - meanBinFreq)**2) / m
	CDelta = (2.0 * meanBinFreq - ssDevBinFreq) / (delta * delta)
	return(m, middleY, lowY, CDelta)

def question2 (data):
	'''
		  Parameter
		  data   : Pandas dataframe.

	 '''

	# Get the values of x that have a group = 1 and 0
	data_g0 = data[data['group'] == 0]['x']
	data_g1 = data[data['group'] == 1]['x']

	# Calculate the IQR as Q3 - Q1
	IQR_g0 = np.percentile(data_g0,75) -  np.percentile(data_g0,25)
	IQR_g1 = np.percentile(data_g1,75) -  np.percentile(data_g1,25)

	# Calculate the Lower whisker for both
	lower_whisker_g0 =  np.percentile(data_g0,25) - 1.5 * IQR_g0
	upper_whisker_g0 =  np.percentile(data_g0,75) + 1.5 * IQR_g0

	# Calculate the upper whisker for both
	lower_whisker_g1 =  np.percentile(data_g1,25) - 1.5 * IQR_g1
	upper_whisker_g1 =  np.percentile(data_g1,75) + 1.5 * IQR_g1

	print('a) The five-number summary of x for each category of the group is: ')
	print('group = 0 -> Min: ' + str(min(data_g0)) + ', Q1: ' + str(np.percentile(data_g0,25)) + ', Q2: ' + str(np.percentile(data_g0,50)) + ', Q3: ' + str(np.percentile(data_g0,75)) + ' and max: ' + str(max(data_g0)))
	print('group = 1 -> Min: ' + str(min(data_g1)) + ', Q1: ' + str(np.percentile(data_g1,25)) + ', Q2: ' + str(np.percentile(data_g1,50)) + ', Q3: ' + str(np.percentile(data_g1,75)) + ' and max: ' + str(max(data_g1)))

	print('\nAnd the values of the 1.5 IQR whiskers are: ')
	print('group = 0 -> lower whisker = ' + str(np.round(lower_whisker_g0,decimals = 2)) + ' and the upper whisker: ' + str(np.round(upper_whisker_g0, decimals=2)))
	print('group = 1 -> lower whisker = ' + str(np.round(lower_whisker_g1,decimals = 2)) + ' and the upper whisker: ' + str(np.round(upper_whisker_g1, decimals=2)))

	# Horizontal Boxplot of the x-values, x (group = 0) and x (group =1)
	fig, ax = plt.subplots(figsize=(10,10))
	ax.set_title('Box Plot')
	ax.boxplot([data['x'], data_g0, data_g1], labels = ['x', 'x (group = 0)', 'x (group = 1)'], vert=False)
	ax.grid(linestyle = '--', linewidth = 0.25)
	plt.show()

	# Get IQR, lower and upper whiskers for all x to get the outliers
	data_g = data['x']

	IQR_g = np.percentile(data_g,75) -  np.percentile(data_g,25)

	lower_whisker_g =  np.percentile(data_g,25) - 1.5 * IQR_g
	upper_whisker_g =  np.percentile(data_g,75) + 1.5 * IQR_g

	print('The five-number summary of x: ')
	print('Min: ' + str(min(data_g)) + ', Q1: ' + str(np.percentile(data_g,25)) + ', Q2: ' + str(np.percentile(data_g,50)) + ', Q3: ' + str(np.percentile(data_g,75)) + ' and max: ' + str(max(data_g)))


	# Print out all the outliers for the three datasets
	print('\nOutliers of x for the entire data:')
	print(data_g[(data_g < lower_whisker_g) | (data_g > upper_whisker_g)])
	
	print('\nOutliers of x for the group = 0:')
	print(data_g0[(data_g0 < lower_whisker_g0) | (data_g0 > upper_whisker_g0)])


	print('\nOutliers of x for the group = 1:')
	print(data_g1[(data_g1 < lower_whisker_g1) | (data_g1 > upper_whisker_g1)])

def question3 (df):
    '''
         Parameter
          df: Pandas dataframe

     '''
    # Count the total number of frauds normalized by the total number of invesitgations
    t = df['FRAUD'].value_counts(normalize = True)
    print('a) ' + str(np.round(t[1]*100, decimals=4)) + ' % of the investigations are a fraud')

    # New dataset with important columns for analysis and transformation to matrix
    df_inter = df[['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION', 'OPTOM_PRESC','NUM_MEMBERS']]
    matrix = np.matrix(df_inter)
    print("Number of Dimensions = ", matrix.ndim)

    xtx = np.dot(matrix.T,matrix)

    # Get eigenvalues and eigenvectors and just consider the ones greater than one
    evals, evecs = np.linalg.eigh(xtx)
    print("Eigenvalues of x = \n", evals)

    evals_1 = evals[evals > 1.0]
    evecs_1 = evecs[:,evals > 1.0]

    print("Eigenvalues of x gretaer than one = \n", evals_1)

    # Get transformation matrix to transform the data
    dvals = 1.0 / np.sqrt(evals_1)
    transf = evecs_1 * np.diagflat(dvals)
    print("Transformation Matrix = \n", transf)

    # Print the input values transformed
    transf_matrix = matrix * transf
    print("The Transformed x = \n", transf_matrix)

    # Check the orthonormalization of transformed X
    xtx = transf_matrix.T * transf_matrix
    print("Expect an Identity Matrix = \n", np.round(xtx,decimals=1))

    Y = df['FRAUD']
    X = transf_matrix

    # Get the KNeighbors model, train it and check the score value.
    model = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')

    res = model.fit(X,Y)
    preds = res.predict(X)
    print('\nc) The score function returns the value: ')
    print(res.score(X,Y))

    # Get a new value and transform it
    x = np.matrix([7500, 15, 3,127,2,2])
    x_t = x * transf
    print('\nd) The new value transformed is: ')

    print(x_t)

    # Check the 5 KNeighbors of the new investigation
    n_5 = model.kneighbors(x_t,return_distance=False)
    print(n_5)
    print(df.iloc[n_5[0]])

    # Predict the probability of classification
    x_f  = res.predict_proba(x_t)

    print(x_f)
    print('The predicted value is of FRAUD with a probability of: ' + str(x_f[0][1]*100) + '%')


def question4 (d_a):
	'''
         Parameter
          d_a: Pandas dataframe of flights (generated by me with the table)
          		provided in the assignment.

     '''
    # Plot a scatterplot of Airport 2 vs Airport 3
	plt.scatter(d_a['Airport 2'], d_a['Airport 3'])
	plt.xlabel('Airport 2')
	plt.ylabel('Airport 3')
	plt.show()

	# Generate a frequency table of the Airport 2 and Airport 3 combined.
	d_air = d_a[['Airport 2','Airport 3']]
	d_23 = pd.concat([d_air['Airport 2'], d_air['Airport 3']])
	freq_tab = d_23.value_counts()
	print(freq_tab)

	# Generation of a vector with the unique airports and a vector for the new flight
	Airports = ['LAX','___', 'LHR', 'EWR','HKG','SFO','VIE','IAD','LAS','LGA','DEN','SEA','CAN','DCA','ICN']
	new_f = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

	# Get the cosine distance between the new flight and all the flights 
	cos_dis = []
	
	for i in range(len(d_air)):
		# Vector of zeros of length = unique airports
		flight = np.zeros(len(freq_tab))
		# Check the value of Airport 2 and 3 of each row of dataset and add 1 in the corresponding position of the vector
		flight_airports= d_air.iloc[i]
		flight[Airports.index(flight_airports['Airport 2'])]=1
		flight[Airports.index(flight_airports['Airport 3'])]=1
		# Calculate the cosine distance
		cosine_distance = 1 -  np.inner(flight, new_f) / (np.linalg.norm(new_f) * np.linalg.norm(flight))
		# Save all distances in the vector cos_dis
		cos_dis.append(cosine_distance)
	
	# Print all distances and the information of the flights with the minimum distance 	
	print('c) the distances from the new flight to all the flights are:')
	print(np.round(cos_dis,decimals=2))
	print('\nd) and the minimum distances correspond to the following flights:')
	print(d_a.iloc[np.where(cos_dis == min(cos_dis))])

### The real work here!
print('-- Question 1 --')

question1(data)

print('-- Question 2 --')

question2(data)

print('-- Question 3 --')

question3(df)

print('-- Question 4 --')

question4(d_a)
