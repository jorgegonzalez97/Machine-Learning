# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics

bikeshare = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\BikeSharingDemand_Train.csv',
                            delimiter=',')

# Use only these four interval variables
trainData = bikeshare[['temp', 'humidity', 'windspeed']].dropna()
nObs = trainData.shape[0]

plt.hist(trainData['temp'], bins = [0,5,10,15,20,25,30,35,40], align = 'mid', color = 'red')
plt.xlabel('Hourly Temperature in Celsius')
plt.xticks(numpy.arange(0,45,5))
plt.yticks(numpy.arange(0,2750,250))
plt.grid(axis = 'y')
plt.show()

plt.hist(trainData['humidity'], bins = [0,10,20,30,40,50,60,70,80,90,100], align = 'mid', color = 'royalblue')
plt.xlabel('Relative Humidity in Percent')
plt.xticks(numpy.arange(0,110,10))
plt.yticks(numpy.arange(0,2250,250))
plt.grid(axis = 'y')
plt.show()

plt.hist(trainData['windspeed'], bins = [0,5,10,15,20,25,30,35,40,45,50], align = 'mid', color = 'green')
plt.xlabel('Windspeed in km/h')
plt.xticks(numpy.arange(0,55,5))
plt.yticks(numpy.arange(0,4000,500))
plt.grid(axis = 'y')
plt.show()

# Determine the number of clusters using the Silhouette metrics
nClusters = numpy.zeros(15)
Elbow = numpy.zeros(15)
Silhouette = numpy.zeros(15)

for c in range(15):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=60616).fit(trainData)

   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]

print("Cluster Size  Elbow Value   Silhouette Value: /n")
for c in range(15):
   print(nClusters[c], Elbow[c], Silhouette[c])

plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.xticks(range(1,16,1))
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.show()

# Plot the Silhouette metrics versus the number of clusters
plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.xticks(range(1,16,1))
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.show()

KClusters = 2
kmeans = cluster.KMeans(n_clusters=KClusters, random_state=60616).fit(trainData)

nC = numpy.zeros(KClusters)
for i in range(nObs):
   k = kmeans.labels_[i]
   nC[k] += 1
print(nC)

for k in range(KClusters):
   print("Cluster ", k)
   print("Centroid = ", kmeans.cluster_centers_[k])

# Load the TREE library from SKLEARN
from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=60616)

bikeshare_DT = classTree.fit(trainData, kmeans.labels_)
print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(trainData, kmeans.labels_)))

import graphviz
dot_data = tree.export_graphviz(bikeshare_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['temp', 'humidity', 'windspeed'],
                                class_names = ['Cluster 0', 'Cluster 1'])

graph = graphviz.Source(dot_data)
graph

graph.render('C:\\IIT\\Machine Learning\\Job\\bikeshare_DT_output')
