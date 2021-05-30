import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RectificationTransformer(nn.Module):

    def __init__(self, imageSize, numOfControlPoints = 10):
        super().__init__()

        self.targetWidth, self.targetHeight = imageSize
        self.numOfControlPoints = numOfControlPoints

        outputControlPoints = self.getOutputControlPoints()

        self.inverseDeltaC = self.getInverseDelta(outputControlPoints)
        self.extendedTargetP = self.getExtendedTargetP(outputControlPoints)

    def forward(self, img, controlPoints):
        
        batchSize = controlPoints.size(0)
    
        expandedControlPoints = torch.cat(
            [
                controlPoints,
                torch.zeros(2, 3).expand(batchSize, 2, 3).float().to(device)
            ], 2)

        T = torch.matmul(expandedControlPoints, self.inverseDeltaC)
        sourceP = torch.matmul(T, self.extendedTargetP)

        grid = sourceP.view(-1,2,self.targetHeight,self.targetWidth).permute(0,2,3,1)
        grid = torch.clip(grid, 0, 1)
        grid = 2 * grid - 1

        rectificatedImg = F.grid_sample(img, grid, align_corners=True)

        return rectificatedImg

    def getOutputControlPoints(self):
        # vraci zakladni pozice kontrolnich bodu pro orezani a zarovnani obrazku
        # vstupem (pres self) je celkovy pocet kontrolnich bodu

        interval = np.linspace(0.05, 0.95, self.numOfControlPoints // 2)
        controlPoints = [[],[]]

        for y in [0.05, 0.95]:
            for i in range(self.numOfControlPoints // 2):
                controlPoints[1].append(y)
            for x in interval:
                controlPoints[0].append(x)

        controlPoints = np.array(controlPoints)

        return torch.Tensor(controlPoints).float().to(device)

    def getRbfKernelMatrix(self, inputPoints, controlPoints):
        # vraci matici obsahujici radialni bazove jadro (RBF kernel) nad euklidovskymi 
        # vzdalenostmi mezi vstupnimi body inputPoints a kontrolnimi body controlPoints

        # vypocet euklidovske vzdalenosti
        distances = []

        for i in range(inputPoints.size(1)):

            cpDistances = []

            for j in range(controlPoints.size(1)):
                diff = inputPoints[...,i] - controlPoints[...,j]
                diff2 = diff * diff
                dist = diff2[0] + diff2[1]
                cpDistances.append(dist)

            distances.append(cpDistances)

        # prevedeni na tensor
        distances = torch.Tensor(distances).float().to(device)

        # vypocet RBF
        rbf = distances * torch.log(distances) * 0.5    # log(sqrt(x)) = log(x) / 2

        # nahrazeni nan hodnot za 0
        mask = torch.isnan(rbf)
        rbf.masked_fill_(mask, 0)

        return rbf

    def getInverseDelta(self, controlPoints):
        # vraci rozsirenou inverzni matici kontrolnich bodu

        numOfControlPoints = self.numOfControlPoints

        # rbf kernel matice mezi kontrolnimi body
        rbfDist = self.getRbfKernelMatrix(controlPoints, controlPoints)

        # vytvoreni tensoru o novych rozmerech
        deltaC = torch.tensor((), dtype=torch.float).float().to(device)
        deltaC = deltaC.new_zeros((numOfControlPoints + 3, numOfControlPoints + 3))
        
        for i in range(0, numOfControlPoints):

            # prekopirovani rbfDist
            for j in range(0, numOfControlPoints):
                deltaC[i + 3, j] = rbfDist[i, j]

            # rozsireni o matice jednicek a o matice kontrolnich bodu
            deltaC[0, i] = 1
            deltaC[1, i] = controlPoints[0, i]
            deltaC[2, i] = controlPoints[1, i]
            deltaC[i + 3, numOfControlPoints] = 1
            deltaC[i + 3, numOfControlPoints + 1] = controlPoints[0, i]
            deltaC[i + 3, numOfControlPoints + 2] = controlPoints[1, i]

        return torch.inverse(deltaC).float().to(device)

    def getExtendedTargetP(self, controlPoints):
        # vraci rozsirenou matici bodu p pro vypocet p' (pro kazdy bod v obrazku)

        targetP = [[],[]]

        # vytvoreni vsech moznych souradnic
        for i in range(self.targetHeight):
            for j in range(self.targetWidth):
                targetP[1].append(i / (self.targetHeight - 1))
                targetP[0].append(j / (self.targetWidth - 1))                

        targetP = torch.Tensor(targetP).float().to(device)
        
        return torch.cat(
            [
                torch.ones(1, self.targetHeight * self.targetWidth).float().to(device),
                targetP,
                self.getRbfKernelMatrix(controlPoints, targetP)
            ], dim = 0)
