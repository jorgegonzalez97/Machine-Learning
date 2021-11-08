import numpy
from numpy import linalg as LA

# Input the matrix X
x = numpy.matrix([[5.1, 160, 82000],
                  [5.2, 170, 84000],
                  [5.3, 180, 86000],
                  [5.4, 190, 88000],
                  [5.5, 200, 90000],
                  [5.6, 110, 81000],
                  [5.7, 120, 83000],
                  [5.8, 130, 85000],
                  [5.9, 140, 87000],
                  [6.0, 150, 89000]])

print("Input Matrix = \n", x)

print("Number of Dimensions = ", x.ndim)

print("Number of Rows = ", numpy.size(x,0))
print("Number of Columns = ", numpy.size(x,1))

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
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

# Orthonormalize using the orth function 
import scipy
from scipy import linalg as LA2

orthx = LA2.orth(x)
print("The orthonormalize x = \n", orthx)

# Check columns of the ORTH function
check = orthx.transpose().dot(orthx)
print("Also Expect an Identity Matrix = \n", check)