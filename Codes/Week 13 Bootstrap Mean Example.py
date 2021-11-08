import matplotlib.pyplot as plt
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

# Calculate the means for the bootstrap samples
def bootstrap_mean (inData, nB):
   n = len(inData)
   outMean = numpy.empty((nB,1))
   for iB in range(nB):
      bootSample = sample_wr(inData)
      bootMean = 0
      for j in range(n):
          bootMean = bootMean + bootSample[j]
      outMean[iB] = bootMean / n
   return outMean

# Summarize the bootstrap results
def summarize_bootstrap (bootResult):
    print('Bootstrap Statistics:\n')
    print('                 Number:', len(bootResult))
    print('                   Mean: {:.7f}' .format(bootResult.mean()))
    print('     Standard Deviation: {:.7f}' .format(bootResult.std()))
    print('95% Confidence Interval: {:.7f}, {:.7f}'
          .format(numpy.percentile(bootResult, (2.5)), numpy.percentile(bootResult, (97.5))))
    
    plt.hist(bootResult, align = 'mid', bins = 50)
    plt.grid(axis='both')
    plt.show()

# Calculate the 95% confidence interval based on the normal distribution
def zConfInterval (inData):
    n = len(inData)
    obsMean = inData.mean()
    obsSE = inData.std() / numpy.sqrt(n)
    z = 1.9599639845
    lowerCI = obsMean - z * obsSE
    upperCI = obsMean + z * obsSE
    print('Observed Mean: {:.7f}' .format(obsMean))
    print('Observed Standard Error: {:.7f}' .format(obsSE))
    print('95% z-confidence interval {:.7f}, {:.7f}:' .format(lowerCI, upperCI))
    
nPopulation = 100

initSeed = 20191113

# Generate X from a Normal with mean = 10 and sd = 7
random.seed(a = initSeed)
population = numpy.zeros((nPopulation,1))
for i in range(nPopulation):
    population[i] = random.gauss(10,7)
zConfInterval (population)
    
meanBoot = bootstrap_mean(population, 500)
summarize_bootstrap(meanBoot)

meanBoot = bootstrap_mean(population, 1000)
summarize_bootstrap(meanBoot)

# Generate X from a Gamma distribution with mean = 10.5
random.seed(a = initSeed)
population = numpy.zeros((nPopulation,1))
for i in range(nPopulation):
    population[i] = random.gammavariate(10.5, 1.0)
zConfInterval (population)

meanBoot = bootstrap_mean(population, 500)
summarize_bootstrap(meanBoot)

meanBoot = bootstrap_mean(population, 1000)
summarize_bootstrap(meanBoot)

# Generate X from a table distribution
random.seed(a = initSeed)
population = numpy.zeros((nPopulation,1))
for i in range(nPopulation):
    u = random.random()
    if (u <= 0.3):
        x = 1
    elif (u <= 0.4):
        x = 2
    elif (u <= 0.8):
        x = 3
    else:
        x = 4
    population[i] = x
zConfInterval (population)

meanBoot = bootstrap_mean(population, 500)
summarize_bootstrap(meanBoot)

meanBoot = bootstrap_mean(population, 1000)
summarize_bootstrap(meanBoot)
