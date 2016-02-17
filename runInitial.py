import readData
import pandas as pd
import DNApy
import numpy as np
import re
import os

def createDimerCount(sequence):

    """
    This is to convert the sequence to 16 features of dimer counts.
    :return: finalCount (NumPy array) - Returns the count of the 16 dimers.

    """

    nucleotides = "ATGC"

    finalCount = []

    # Finding counts of overlapping matches using a positive lookahead assertion.
    for nucleotide1 in nucleotides:
        for nucleotide2 in nucleotides:
            dimerCount = len(list(re.finditer(r"%s(?=%s)" % (nucleotide1, nucleotide2), sequence)))
            finalCount.append(dimerCount)

    # Returning the counts of the 16 dimers.
    return np.array(finalCount)


class CreateFiles(object):
    def __init__(self):
        self.data = readData.readData( readData.datafilePath )


    def createFileOfGeneNames(self):
        """
        This creates the file to upload to Ensembl ( Biomart ) for
        querying gene location.
        """
        openfile = open("geneNames.txt", "w")
        for i in set( self.data[:, 0] ):
            openfile.write(i + "\n")

        openfile.close()

    def getHairpinCount(self, oligonucleotide):
        """
        Returns the number of hairpins.

        There are a few assumptions:
        1. No mismatches allowed.
        2. Loop length can only be between 4 and 7.
        3. Hairpin length can only be between 3 and 6.
        4. Melting temperature of hairpin is much higher than 37 degrees.

        :param oligonucleotide (string) - The oligonucleotide.
        :return Number of hairpins (int).
        """

        oligoLength = len(oligonucleotide)
        count = 0
        loopLengths = range(4, 8) # Loops with less than 4 bases are not sterically possible.

        # Length of loop.
        for loopLength in loopLengths:

            # Length of hairpin.
            for hairpinLength in range(3, 7):

                # Iterating through each position.
                for j in range(oligoLength - (2*hairpinLength+loopLength-2)):
                    oneSide = oligonucleotide[j:j+hairpinLength]
                    otherSide = oligonucleotide[j+hairpinLength+4:j+2*hairpinLength+4]
                    if oneSide == DNApy.revcompl(otherSide):
                        count += 1

        return count

    def makeHairpinCountFile(self):
        """
        Creates the file containing hairpin counts for each oligonucleotide
        in the dataset.
        """

        # Fetching the oligonucleotide column.
        oligonucleotides = self.data[:, 2]
        hairpinCounts = map(self.getHairpinCount, oligonucleotides)

        openfile = open( readData.hairpinFileName, "w" )
        for i in hairpinCounts:
            openfile.write("%s\n" % i)
        openfile.close()


    def createDimerCountsFile(self):
        """
        This is to convert the sequence to 16 features of dimer counts.
        :return: Returns the count of the dimers (list).

        """
        sequences = self.data[:, 2]
        finalCount = np.array([createDimerCount(i) for i in sequences])
        np.save(readData.dimerCountsFileName, finalCount)
        return finalCount


if __name__ == "__main__":

    createFiles = CreateFiles()

    # Creating the gene names file.
    createFiles.createFileOfGeneNames()

    # Creating the hairpin counts file.
    print 'creating hairpin counts file'
    createFiles.makeHairpinCountFile()
    print 'done'

    # Creating the dimer counts file.
    print 'creating dimer counts file'
    dimerCounts = createFiles.createDimerCountsFile()

    os.system("touch %s" % readData.runInitialFileName)