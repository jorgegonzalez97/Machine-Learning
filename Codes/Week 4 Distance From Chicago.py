import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas

DistanceFromChicago = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\DistanceFromChicago.csv',
                      delimiter=',', index_col='CityState')

nCity = DistanceFromChicago.shape[0]

trainData = numpy.reshape(numpy.asarray(DistanceFromChicago['DrivingMilesFromChicago']), (nCity, 1))

# Determine the number of clusters
maxNClusters = 15

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
Silhouette = numpy.zeros(maxNClusters)
Calinski_Harabasz = numpy.zeros(maxNClusters)
Davies_Bouldin = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (1 < KClusters):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
       Calinski_Harabasz[c] = metrics.calinski_harabasz_score(trainData, kmeans.labels_)
       Davies_Bouldin[c] = metrics.davies_bouldin_score(trainData, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN
       Calinski_Harabasz[c] = numpy.NaN
       Davies_Bouldin[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCity):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData[i] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

plt.plot(nClusters, TotalWCSS, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Total WCSS")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()   

plt.plot(nClusters, Calinski_Harabasz, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Davies_Bouldin, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()   

# Display the 4-cluster solution
kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

# ClusterResult = DistanceFromChicago
# ClusterResult['ClusterLabel'] = kmeans.labels_

cmap = ['indianred','sandybrown','royalblue', 'olivedrab']

fig, ax = plt.subplots()
for c in range(4):
   subData = DistanceFromChicago[kmeans.labels_ == c]
   plt.hist(subData['DrivingMilesFromChicago'], color = cmap[c], label = str(c), linewidth = 2, histtype = 'step')
ax.set_ylabel('Number of Cities')
ax.set_xlabel('DrivingMilesFromChicago')
ax.set_xticks(numpy.arange(0,2500,250))
plt.grid(axis = 'y')
plt.legend(loc = 'lower left', bbox_to_anchor = (0.15, 1), ncol = 4, title = 'Cluster ID')
plt.show()

for c in range(4):
    print("Cluster Label = ", c)
    print(DistanceFromChicago[kmeans.labels_ == c])

