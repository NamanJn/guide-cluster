import pandas as pd
from os.path import join
import os, re, sys
import numpy as np
from DNApy.clustering import ClusterMotifs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.stats.stats import pearsonr

# File Names.
fileName = "example_guide_data.tsv"
hairpinFileName = 'hairpinCounts.txt'
biomartResultsFileName = "biomartResults.txt"
runInitialFileName = "runinitial.txt"
dimerCountsFileName = "dimerCounts.npy"

dataDir = "data"
datafilePath = join(dataDir, fileName)

nucleotideWeights = { "A": 347.2, "C": 323.2, "G": 363.2, "T": 324.2 }



def readData(datafilePath):
    """
    Reading the data.

    :param datafilePath (string) - The relative path to the data file.
    :return: d (NumPy array) - Contents of the file.
    """
    # reading the data
    data = pd.read_table(datafilePath, sep="\t", header=0)
    d = data.values
    return d


def readChromosomeLengthFile():
    """
    Reading the 'chromosome lengths' file.

    :return: chromosome length data (dict: key - chromosome, value - length of chromosome).
    """

    openfile = open( "chromosomeSizes.txt" )
    readfile = openfile.read()
    openfile.close()
    return dict([ re.split(r"\s+", i) for i in readfile.strip().split("\n") ])


def clusterThem(spacers):
    """
    Clustering the spacers using k-means.

    :param spacers (list) - The array of oligonucleotides.
    :return: groups (dictionary) - The oligonucleotides grouped into clusters.
    """
    cluster = ClusterMotifs(3)
    groups = cluster.clusterMotifs(spacers)
    return groups


def visualise(number, bins=20):
    sns.distplot(number, bins=bins)
    sns.plt.savefig("histogram.png")
    os.system("open histogram.png")


def getRatio(data):
    """
    Finding the day1/day14 ratio.

    :param data (NumPy array) - The data present in example_guide_data.tsv
    :returns ratio (NumPy array) - The array.
    """
    # Day-14 average of both replicates.
    day14avg =  ( data[:, 4] + data[:, 5] )/ 2

    # Day-1/Day-14 average.
    ratio = data[:, 3]/day14avg
    return ratio.astype("float64")


def getEssentialGenes(d):
    """
    Gets the most important genes by using the day1/day14 ratio.
    Returns a list of tuples where the tuples contain the gene and the ratio.

    :param d (NumPy array) - The raw data matrix.
    :return: List of tuples (list). Tuples contains 2 elements.
    """

    # Getting the depleted ratio for each spacer.
    dep = getRatio( d )
    names = d[ :, 0 ]
    combined =  np.vstack([dep, names]).T
    lol = [ list(i) for i in combined ]

    # Aggregating the data for each gene.
    dicto = {}
    for i in lol:
        key = i[1]
        dicto[key] = dicto.get(key, []) + [i[0]]

    # Finding the average depleted ratio per gene.
    averages = {}
    for i in dicto.keys():
        averages[i] = np.average(dicto[i])

    # Sorting ( descending ) the genes according to their average depleted ratio.
    essential = sorted(averages.items(), key=lambda x: x[1], reverse=True)

    return essential


def getMeltingTemperature(oligonucleotide):
    """
    Using the Wallace Rule to find the melting temperature of an oligonucleotide.

    :param oligonucleotide (string) - The oligonucleotide.
    :returns Melting temperature (int).
    """
    gcCount = oligonucleotide.count("G") + oligonucleotide.count("C")
    atCount = len(oligonucleotide) - gcCount
    return gcCount*4 + atCount*2


def getMass(oligonucleotide):
    """
    Returns the mass of the oligonucleotide.

    :param oligonucleotide (string) - The oligonucleotide.
    """
    return sum([nucleotideWeights[i] for i in oligonucleotide])


def readBioMartResults():
    """
    Reads the gene locations results from BioMart.

    :return: positionMapper, chromosomeMapper (tuple) -
    """
    a = pd.read_csv( biomartResultsFileName, sep="\t")
    positionMapper = {}
    chromosomeMapper = {}

    for i in a.values:
        if len(i[1]) > 4:
            continue
        positionMapper[ i[0] ] = i[3]
        chromosomeMapper[ i[0] ] = i[1]

    return positionMapper, chromosomeMapper


def readHairpinCounts():
    """
    Reads the hairpin file that was created by runInitial
    :return: hairpinCounts (list)
    """
    openfile = open( hairpinFileName )
    hairpinCounts = [int(i) for i in openfile.read().strip().split("\n")]
    openfile.close()
    return hairpinCounts


def createMatrixForClustering(data):
    """
    Returns the NumPy 2D array.
    Column = Features.
    Row = Data.

    The features are:
    1. Melting temperature.
    2. Mass.
    3. Position in genome.
    4. Hairpin count.
    5. Sequence (as 16 features , 2-mers ).

    :param data (NumPy array) - The data present in example_guide_data.tsv.
    :return feature matrix (NumPy array) - The feature matrix.
    """

    oligonucleotides = data[:, 2]

    position_mapper, chromosome_mapper = readBioMartResults()
    hairpinCounts = readHairpinCounts()
    sizes = readChromosomeLengthFile()
    dimerCounts = np.load(dimerCountsFileName)

    results = []

    # Iterating through the rows.
    for index, i in enumerate(data):
        oligonucleotide = i[2]

        # Getting the melting temperature.
        meltingTemperature = getMeltingTemperature(oligonucleotide)

        # Getting the mass of the oligonucleotide.
        mass = getMass(oligonucleotide)

        # Getting the relative position of the gene.
        position = position_mapper.get(i[0], 300000)/float(sizes[ 'chr%s' % chromosome_mapper.get( i[0], 1) ])

        # Getting the hairpin count.
        hairpinCount = int(hairpinCounts[index])

        results.append([meltingTemperature, mass, position, hairpinCount]+list(dimerCounts[index]))

    return np.array(results)


def reduceDimensionalityToTwo(matrix):
    """
    Reduces the dimension via PCA (to 2) for visualisation.

    :param matrix (NumPy array) - The feature matrix.
    :return: coordinates (NumPy array) - Array containing the x and y coordinates.
    """
    pca = PCA()
    pca.n_components = 2
    coordinates = pca.fit_transform(matrix)

    return coordinates


def createPlot(matrix, ratios, figName, xlabel, ylabel):
    """
    Creates a scatter plot.

    :param matrix (NumPy Array) - The coordinates.
    :param ratios (NumPy Array) - The guide activity ratios.
    :param figName (string) - The figure name including the file extension type.

    """

    x = matrix[:, 0]
    y = matrix[:, 1]

    if type(ratios) == int:
        plt.scatter(x, y,  alpha=1)
    else:
        plt.scatter(x, y, c=list(ratios), alpha=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figName)
    plt.clf()


def clusterData(matrix, numOfClusters):
    """

    :param matrix (NumPy array) - The feature-scaled data matrix.
    :param numOfClusters (int) - The number of clusters
    :return: labels (NumPy array) - The group number for each sample.
    """
    kmeans = KMeans(n_clusters=numOfClusters)
    labels = kmeans.fit(matrix).labels_
    return labels


def getCorrelationHairpin():
    """
    Plotting the scatter plot and regression line of the hairpinCount
    vs the guide activities.
    """

    # Getting the guide activity.
    ratios = np.log2(getRatio(d))
    # Getting the hairpinCounts.
    hairpinCounts = np.array(readHairpinCounts()).astype("float64")

    # Plotting the scatterplot and regression line.
    sns.regplot(hairpinCounts, ratios)

    sns.plt.xlabel("Hairpin Count")
    sns.plt.ylabel("Guide Activity")
    sns.plt.savefig('hairpinVsGuide.png')
    sns.plt.clf()


def getCorrelationMeltingTemperature(d):
    """
    Plotting the scatter plot and regression line of guide melting temperature
    vs the guide activities.
    """

    # Getting the guide activity.
    ratios = np.log2(getRatio(d))

    # Getting the hairpinCounts.
    oligonucleotides = d[:, 2]
    meltingTemperatures = np.array([ getMeltingTemperature(oligonucleotide) for oligonucleotide in oligonucleotides ])

    # Plotting the scatterplot and regression line.
    sns.regplot(meltingTemperatures, ratios)

    sns.plt.xlabel("Melting Temperature")
    sns.plt.ylabel("Guide Activity")
    sns.plt.savefig('TemperaturesVsGuide.png')
    sns.plt.clf()
    return pearsonr(meltingTemperatures, ratios)


def getCorrelationGenomeLocation():
    """
    Plotting the scatter plot and regression line of the genomeLocation
    vs the guide activities.

    """
    ratios = np.log2(getRatio(d))

    # Getting the hairpinCounts.
    positionMapper, chromosomeMapper = readBioMartResults()

    sizes = readChromosomeLengthFile()
    data = readData(datafilePath)

    allPositions = []

    for index, i in enumerate(data):

        chromosomeLength = sizes[ 'chr%s' % chromosomeMapper.get( i[0], 1) ]
        position = positionMapper.get(i[0], 300000)/float(chromosomeLength)

        if position > 1:
            pass

        allPositions.append(position)

    allPositions = np.array(allPositions).astype("float64")

    # Plotting the scatterplot and regression line.
    sns.regplot(allPositions, ratios, fit_reg=False)
    sns.plt.xlabel("Genome location ( position / chromosome length)")
    sns.plt.ylabel("Guide activity")
    sns.plt.savefig('positionVsGuide.png')
    sns.plt.clf()

def featureScaleMatrix(matrix):
    """
    This function scales the features (between 0 and 1).
    It also reduces the weight of the 16-features generated by
    the guide sequence.

    :param matrix (NumPy array) - The feature matrix.
    :return: normalised matrix (NumPy array) - The feature matrix post-normalisation.
    """

    # feature scaling
    minmax = preprocessing.MinMaxScaler()
    normalisedMatrix = minmax.fit_transform(matrix)

    # dividing the normalised values for dimer-features by 16
    normalisedMatrix[:, 4:] = normalisedMatrix[:, 4:]/16.0

    return normalisedMatrix


if __name__ == "__main__":

    if not os.path.exists(runInitialFileName):
        print "\n\nPlease run runInitial.py first"
        sys.exit()

    # Reading in the data
    d = readData(datafilePath)

    # Getting the guide activity ratios.
    ratios = np.log2(getRatio(d))

    # Getting the oligonucleotides.
    oligonucleotides = d[:, 2]

    # Testing correlation between genome location and guide activity.
    getCorrelationGenomeLocation()

    # Testing correlation between hairpin count and guide activity.
    getCorrelationHairpin()

    # Testing correlation between Melting temperature and guide activity.
    tempCorr = getCorrelationMeltingTemperature(d)

    # Getting the essential genes.
    essential = getEssentialGenes(d)

    # Printing the 10 most essential genes.
    print "essential genes\n", [i[:] for i in essential[:10]]

    # Creating the feature matrix.
    matrix = createMatrixForClustering(d)

    # Feature scaling between 0 and 1.
    normalisedMatrix = featureScaleMatrix(matrix)

    # Performing dimensionality reduction via PCA for visualisation.
    coordinates = reduceDimensionalityToTwo(normalisedMatrix)

    # Plotting the output of PCA.
    figName = "pcaOutput.png"
    createPlot(coordinates, ratios, figName, "pc1", "pc2")

    # Clustering the guides.
    numberOfClusters = 4
    km = clusterData(normalisedMatrix, numberOfClusters)

    # Getting the average guide activity for each cluster.
    figName = "guideActivitiesForEachCluster.png"
    coordinates = []
    plotRatios = np.array([])
    for clusterNumber in range(numberOfClusters):
        coo = km == clusterNumber
        averageRatio = np.average(ratios[coo])
        coordinates += zip( [clusterNumber]*len(ratios[coo]), list(ratios[coo]) )
        print averageRatio

    # Also plotting guide activity distribution for each cluster.
    createPlot(np.array(coordinates), 0, figName, "Cluster Number", "Guide activities")



