import matplotlib.pyplot as plt
import numpy
import pandas

inData = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\Column_Y.csv')
Y = inData['Y']
print(Y.describe())

plt.hist(Y)
plt.show()

def calcCD (Y, delta):
   maxY = numpy.max(Y)
   minY = numpy.min(Y)
   meanY = numpy.mean(Y)

   # Round the mean to integral multiples of delta
   middleY = delta * numpy.round(meanY / delta)

   # Determine the number of bins on both sides of the rounded mean
   nBinRight = numpy.ceil((maxY - middleY) / delta)
   nBinLeft = numpy.ceil((middleY - minY) / delta)
   lowY = middleY - nBinLeft * delta

   # Assign observations to bins starting from 0
   m = nBinLeft + nBinRight
   BIN_INDEX = 0;
   boundaryY = lowY
   for iBin in numpy.arange(m):
      boundaryY = boundaryY + delta
      BIN_INDEX = numpy.where(Y > boundaryY, iBin+1, BIN_INDEX)

   # Count the number of observations in each bins
   uBin, binFreq = numpy.unique(BIN_INDEX, return_counts = True)

   # Calculate the average frequency
   meanBinFreq = numpy.sum(binFreq) / m
   ssDevBinFreq = numpy.sum((binFreq - meanBinFreq)**2) / m
   CDelta = (2.0 * meanBinFreq - ssDevBinFreq) / (delta * delta)
   return(m, middleY, lowY, CDelta)

result = pandas.DataFrame()
deltaList = [1000, 2000, 5000, 10000, 20000, 50000]

for d in deltaList:
   nBin, middleY, lowY, CDelta = calcCD(Y,d)
   highY = lowY + nBin * d
   result = result.append([[d, CDelta, lowY, middleY, highY, nBin]], ignore_index = True)

   binMid = lowY + 0.5 * d + numpy.arange(nBin) * d
   plt.hist(Y, bins = binMid, align='mid')
   plt.title('Delta = ' + str(d))
   plt.ylabel('Number of Observations')
   plt.grid(axis = 'y')
   plt.show()
   
result = result.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin'})

fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot')
ax1.boxplot(Y, labels = ['Y'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()
