import numpy
import random

# Create a bootstrap sample from the population
def sample_wr (inData):
    n = len(inData)
    outData = numpy.empty((n,1))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

x = numpy.array([1,2,3,4,5,6,7,8,10])
unique, counts = numpy.unique(x, return_counts = True)
print('Original Sample:\n', numpy.asarray((unique, counts)).transpose())

# Check the bootstrap sample
random.seed(20191113)

sample1 = sample_wr(x)
unique, counts = numpy.unique(sample1, return_counts = True)
print('Sample 1:\n', numpy.asarray((unique, counts)).transpose())

sample2 = sample_wr(x)
unique, counts = numpy.unique(sample2, return_counts = True)
print('Sample 2:\n', numpy.asarray((unique, counts)).transpose())

sample3 = sample_wr(x)
unique, counts = numpy.unique(sample3, return_counts = True)
print('Sample 3:\n', numpy.asarray((unique, counts)).transpose())

sample4 = sample_wr(x)
unique, counts = numpy.unique(sample4, return_counts = True)
print('Sample 4:\n', numpy.asarray((unique, counts)).transpose())
