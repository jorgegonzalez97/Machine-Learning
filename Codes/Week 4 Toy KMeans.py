import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics

# Specify as a 2-dimnensional array to meet the KMeans requirements
trainData = numpy.array([[2, 11], [4, 11], [2, 9], [4, 9],
                 [6, 11], [8, 11], [6, 9], [8, 9],
                 [4.5, 5.5], [5.5, 5.5], [5.5, 4.5], [4.5, 4.5]])

plt.scatter(x = trainData[:,0], y = trainData[:,1], marker = 'o')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.xticks(numpy.arange(1, 10, step = 1))
plt.yticks(numpy.arange(3, 13, step = 1))
plt.show()

nObs = trainData.shape[0]

# Determine the number of clusters
maxNClusters = 6

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

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData[i] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   print("Cluster Assignment:", kmeans.labels_)
   for k in range(KClusters):
      print("Cluster ", k)
      print("Centroid = ", kmeans.cluster_centers_[k])
      print("Size = ", nC[k])
      print("Within Sum of Squares = ", WCSS[k])
      print(" ")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value")
for c in range(maxNClusters):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

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
