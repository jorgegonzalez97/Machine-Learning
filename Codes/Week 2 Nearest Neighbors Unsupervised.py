# Load the necessary libraries
import numpy
import pandas
from sklearn.neighbors import NearestNeighbors as kNN

cars = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv', delimiter=',')

cars["CaseID"] = cars["Make"] + "_" + cars.index.values.astype(str)

cars_wIndex = cars.set_index("CaseID")

# Specify the kNN
kNNSpec = kNN(n_neighbors = 4, algorithm = 'brute', metric = 'euclidean')

# Specify the training data
trainData = cars_wIndex[['Invoice', 'Horsepower', 'Weight']]
trainData.describe()

# Build nearest neighbors
nbrs = kNNSpec.fit(trainData)
distances, indices = nbrs.kneighbors(trainData)

# Find the nearest neighbors of these focal observations       
focal = [[173560, 477, 3131],     # Porsche_335
         [119600, 493, 4473],     # Mercedes-Benz_263
         [117854, 493, 4429],     # Mercedes-Benz_272
         [113388, 493, 4235]]     # Mercedes-Benz_271

myNeighbors = nbrs.kneighbors(focal, return_distance = False)
print("My Neighbors = \n", myNeighbors)

# Orthonormalized the training data
x = numpy.matrix(trainData.values)

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = numpy.linalg.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
dvals = 1.0 / numpy.sqrt(evals)
transf = evecs * numpy.diagflat(dvals)
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)

# Specify the kNN
kNNSpec = kNN(n_neighbors = 4, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(transf_x)
distances, indices = nbrs.kneighbors(transf_x)

# Find the nearest neighbors of these focal observations       
focal = [[173560, 477, 3131],     # Porsche_335
         [119600, 493, 4473],     # Mercedes-Benz_263
         [117854, 493, 4429],     # Mercedes-Benz_272
         [113388, 493, 4235]]     # Mercedes-Benz_271

transf_focal = focal * transf;

myNeighbors_t = nbrs.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = \n", myNeighbors_t)
