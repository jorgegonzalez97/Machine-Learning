import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics

# Specify as a 2-dimnensional array to meet the KMeans requirements
X = numpy.array([[0.1], [0.3], [0.4], [0.8], [0.9]])

# Find the 2-cluster solution
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(X)
print("Cluster Assignment:", kmeans.labels_)
print("Cluster Centroid 0:", kmeans.cluster_centers_[0])
print("Cluster Centroid 1:", kmeans.cluster_centers_[1])

# Determine the number of clusters
nClusters = numpy.zeros(5)
Elbow = numpy.zeros(5)
Silhouette = numpy.zeros(5)
TotalWCSS = numpy.zeros(5)
Inertia = numpy.zeros(5)

for c in range(5):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(X)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (1 < KClusters & KClusters < 5):
       Silhouette[c] = metrics.silhouette_score(X, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(5):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X[i] - kmeans.cluster_centers_[k]
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

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(5):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(1, 6, step = 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(1, 6, step = 1))
plt.show()    
