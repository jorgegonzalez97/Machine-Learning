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
def bootstrap_cv (inData, nB):
   n = len(inData)
   outCV = numpy.empty((nB,1))
   for iB in range(nB):
      bootSample = sample_wr(inData)
      bootMean = 0
      for j in range(n):
          bootMean = bootMean + bootSample[j]
      bootMean = bootMean / n
      bootSD = 0
      for j in range(n):
         bootSD = bootSD + (bootSample[j] - bootMean) ** 2
      bootSD = numpy.sqrt(bootSD / (n - 1))
      outCV[iB] = 100 * (bootSD / bootMean)
   return outCV

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
    obsCV = 100 * (inData.std() / inData.mean())
    seCV = (obsCV / numpy.sqrt(2*n)) * numpy.sqrt(1 + 2 * (obsCV / 100)**2)
    z = 1.9599639845
    lowerCI = obsCV - z * seCV
    upperCI = obsCV + z * seCV
    print('Observed CV: {:.7f}' .format(obsCV))
    print('Observed Standard Error: {:.7f}' .format(seCV))
    print('95% z-confidence interval {:.7f}, {:.7f}:' .format(lowerCI, upperCI))

nPopulation = 100
initSeed = 20191113

# Generate X from a Normal with mean = 10 and sd = 7
random.seed(a = initSeed)
population = numpy.zeros((nPopulation,1))
for i in range(nPopulation):
    population[i] = random.gauss(10,7)
zConfInterval (population)
    
cvBoot = bootstrap_cv(population, 500)
summarize_bootstrap(cvBoot)

cvBoot = bootstrap_cv(population, 1000)
summarize_bootstrap(cvBoot)

# Generate X from a Gamma distribution with mean = 10.5
random.seed(a = initSeed)
population = numpy.zeros((nPopulation,1))
for i in range(nPopulation):
    population[i] = random.gammavariate(10.5, 1.0)
zConfInterval (population)

cvBoot = bootstrap_cv(population, 500)
summarize_bootstrap(cvBoot)

cvBoot = bootstrap_cv(population, 1000)
summarize_bootstrap(cvBoot)

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

cvBoot = bootstrap_cv(population, 500)
summarize_bootstrap(cvBoot)

cvBoot = bootstrap_cv(population, 1000)
summarize_bootstrap(cvBoot)
