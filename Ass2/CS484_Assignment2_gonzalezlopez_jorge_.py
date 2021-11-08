##########################################
### CS484 Spring 2021
### Assignment 2
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
import sklearn.cluster as cluster
import sklearn.metrics as metrics

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


### Load the data
df_groc = pd.read_csv('Groceries.csv')

df_cars = pd.read_csv('cars.csv')

### Preprocessing


### Define a function
def question4 ():
    Cluster0 = [-2,-1,1,2,3]
    Cluster1 = [4,5,7,8]
    data = Cluster0 + Cluster1
    data = np.vstack(data)

    a_ij = 0
    d_ij = 0

    for i in Cluster0:
            if i != -1:
               a_ij += (np.linalg.norm(-1 - i) / (len(Cluster0)-1))

    for i in Cluster1:
         d_ij += (np.linalg.norm(-1 - i) / (len(Cluster1)))

    #Only two clusters
    b_ij = d_ij


    s_ij = (b_ij - a_ij) / np.maximum(a_ij, b_ij)

    print('a) The Silhouette width of the observation 2 in Cluster 0 is: ' + str(s_ij))

    cen_0 = np.mean(Cluster0)
    cen_1 = np.mean(Cluster1)

    S_0 = 0
    S_1 = 0

    for i in Cluster0:
        S_0 += (np.linalg.norm(cen_0 - i) / (len(Cluster0)))

    for i in Cluster1:
        S_1 += (np.linalg.norm(cen_1 - i) / (len(Cluster1)))

    M_kl = np.linalg.norm(cen_0 - cen_1) 
    R_kl = (S_0 + S_1)/(M_kl)

    R_0 = np.max(R_kl)
    R_1 = np.max(R_kl)

    print('\b) The Davies-Bouldin value of cluster 0 is : '+ str(np.round(R_0, decimals=4))+ ' and of cluster 1 is: '+ str(np.round(R_1, decimals=4)))

    DB = (R_0 + R_1) / 2

    print('\nc) The Davies-Bouldin Index is: '+ str(np.round(DB, decimals=4)))




def question5 (df):
	'''
		  Parameter
		  df_groc   : Pandas dataframe.

	 '''
	# Get itemSets of all the customers
	itemsets = df.groupby(['Customer'])['Item'].apply(list).values.tolist()

	# Convert the Item List format to the Item Indicator format
	te = TransactionEncoder()
	te_ary = te.fit(itemsets).transform(itemsets)
	item_ind = pd.DataFrame(te_ary, columns=te.columns_)

	support = 75 / len(itemsets)
	# Find the frequent itemsets
	frequent = apriori(item_ind, min_support = support, use_colnames = True)

	print('\na) There are ' + str(len(frequent)) + ' itemsets in total.')

	largest_k = frequent['itemsets'].values[-1]

	print('\nThe larget k value among the itemsets is: ' + str(len(largest_k)))

	# Association rules with a Coincidence metric greater or equal to 1%
	coinc = 0.01
	a_r = association_rules(frequent, metric = "confidence", min_threshold = coinc)

	print('\nb) There are ' + str(len(a_r)) + ' association rules.')

	plt.figure(figsize=(10,7))
	plt.colorbar(plt.scatter(a_r['confidence'], a_r['support'], a_r['lift'], a_r['lift'], linewidths=2), label= 'Lift')
	plt.xlabel("Confidence")
	plt.ylabel("Support")
	plt.grid()
	plt.show()

	# Confidence >= 60 %
	a_r2 = a_r[a_r['confidence'] >= 0.6]
	print('\nd)\n')
	print(a_r2[['antecedents', 'consequents','support', 'confidence', 'lift']])

def question6 (df):
    '''
         Parameter
          df_cars: Pandas dataframe

     '''
    input_var = ['Weight', 'Wheelbase', 'Length']
    k_min = 2
    k_max = 10
    random_state = 60616

    # Keep a copy of values without scaling
    df_nn = df[input_var]

    #Scale values in range [0, 10]
    df[input_var] = 10*(df[input_var]-np.min(df[input_var]))/(np.max(df[input_var]) - np.min(df[input_var]))

    #Get all values for a range of clusters (2 to 10)
    Elbow = []
    Sil = []
    Cal = []
    Dav = []
    wcss = []

    for i in range(k_min, k_max+1):
        km = cluster.KMeans(n_clusters = i, random_state= random_state).fit(df[input_var])

        Sil.append(metrics.silhouette_score(df[input_var], km.labels_))
        Cal.append(metrics.calinski_harabasz_score(df[input_var], km.labels_))
        Dav.append(metrics.davies_bouldin_score(df[input_var], km.labels_))

        WCSS = np.zeros(i)
        nC = np.zeros(i)

        for j in range(df[input_var].shape[0]):
            k = km.labels_[j]
            nC[k] += 1
            diff = df[input_var].values[j] - km.cluster_centers_[k]
            WCSS[k] += diff.dot(diff)

        elb = 0
        TotalWCSS = 0
        for k in range(i):
            elb += WCSS[k] / nC[k]
            TotalWCSS += WCSS[k]

        Elbow.append(elb)
        wcss.append(TotalWCSS)


    print('a)\n')
    print("K\t WCSS\t      Elbow Value \t    Silhouette values \t Calinski-Harabasz Scores\t Davies-Bouldin Index")
    for c in range(k_min, k_max+1):
       print('{:.0f} \t {:.4f} \t {:.4f}  \t {:.4f}            \t {:.4f}                  \t {:.4f}'
             .format(c, wcss[c-2], Elbow[c-2], Sil[c-2], Cal[c-2], Dav[c-2]))

    print('\nb) The suggested number of clusters is: ' + str(3))


    km = cluster.KMeans(n_clusters = 3, random_state= random_state).fit(df_nn)

    print('\nc) Clusters centroids with origin values: ')
    print("Cluster Centroid 0:", km.cluster_centers_[0])
    print("Cluster Centroid 1:", km.cluster_centers_[1])
    print("Cluster Centroid 2:", km.cluster_centers_[2])

### The real work here!
print('-- Question 4 --')

question4()

print('-- Question 5 --')

question5(df_groc)

print('-- Question 6 --')

question6(df_cars)