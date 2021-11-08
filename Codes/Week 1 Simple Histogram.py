import matplotlib.pyplot as plt
import numpy

x = numpy.array([213,214,214,215,216,216,216,217,217,218])

# Generate a frequence table
uvalue, ucount = numpy.unique(x, return_counts = True)
print('Unique Values:\n', uvalue)
print('Unique Counts:\n', ucount)

# Draw a properly labeled histogram with default specification
plt.hist(x)
plt.title('My Weights in Past Ten Days')
plt.xlabel('Weight (lbs)')
plt.ylabel('Number of Days')
plt.show()

# Draw a better labeled histogram with specified bin boundaries
plt.hist(x, bins = [212.5,213.5,214.5,215.5,216.5,217.5,218.5], align='mid')
plt.title('My Weights in Past Ten Days')
plt.xlabel('Weight (lbs)')
plt.ylabel('Number of Days')
plt.yticks(range(4))
plt.grid(axis = 'y')
plt.show()

binMid = 212.5 + numpy.arange(7) * 1
plt.hist(x, bins = binMid, align='mid')
plt.title('My Weights in Past Ten Days')
plt.xlabel('Weight (lbs)')
plt.ylabel('Number of Days')
plt.yticks(range(4))
plt.grid(axis = 'y')
plt.show()
