import os
import sys
import numpy as np
from matplotlib import pyplot as plt

def getPathFromArgs(args):
    if len(args) != 2:
        print("plotMaker.py <inputfile>")
        exit(2)
    
    return args[1]

if __name__ == "__main__":
    
    sourceFilePath = getPathFromArgs(sys.argv)
    outputDirectory = "plots"
    outputFilePath = os.path.join(outputDirectory, "loss.png")
    
    results = np.load(sourceFilePath)

    x = np.arange(len(results[:,0])) + 1
    
    y1 = results[:,0]
    y2 = results[:,1]

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.plot(x, y1, label='Training')
    plt.plot(x, y2, label='Validation')
    
    plt.legend()
    
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
        
    plt.savefig(outputFilePath, format='png')
