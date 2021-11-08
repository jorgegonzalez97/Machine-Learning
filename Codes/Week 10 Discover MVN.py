import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics

trainData = pandas.read_excel('C:\\IIT\\Machine Learning\\Data\\MVN.xlsx',
                              sheet_name = 'MVN', usecols = ['X', 'Y'])

nObs = trainData.shape[0]

# Determine the number of clusters
maxNClusters = 10
nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
Silhouette = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20191106).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (1 < KClusters):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData.iloc[i] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

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

# Fit the 2 cluster solution
kmeans = cluster.KMeans(n_clusters=2, random_state=20191106).fit(trainData)
centroid = kmeans.cluster_centers_
print('Centroids:\n', centroid)

trainData['Cluster ID'] = kmeans.labels_

carray = ['red', 'green']
plt.figure(dpi = 200)
for i in range(2):
    subData = trainData[trainData['Cluster ID'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.scatter(x = centroid[:,0], y = centroid[:,1], c = 'black', marker = 'X', s = 100)
plt.grid(True)
plt.title('2-KMeans Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(numpy.arange(-3,4,1))
plt.yticks(numpy.arange(-3,4,1))
plt.legend(title = 'Cluster ID', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
