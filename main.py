import os
import sys
import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import Levenshtein
import cv2

from PIL import Image
from models.networkArchitecture import NetworkArchitecture 
from inputParser import InputParser
from coder import Coder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imgWidth = 100
imgHeight = 32

def parseArgs(argv):
    # parsovani vstupnich argumentu

    ap = argparse.ArgumentParser()

    ap.add_argument("-b", "--batchSize", required=True, help="Batch size")
    ap.add_argument("-e", "--epochs", required=False, help="Number of epochs")
    ap.add_argument("-o", "--outputFolder", required=False, help="Output folder path")
    ap.add_argument("-t", "--test", required=False, action="store_true", help="Run test instead of training")
    ap.add_argument("-m", "--model", required=False, help="Path to the model to load")
    ap.add_argument("-r", "--rectification", required=False, action="store_true", help="Use image rectification")
    ap.add_argument("-bl", "--baseline", required=False, action="store_true", help="Use baseline recognition network")
    ap.add_argument("-a", "--augmentation", required=False, action="store_true", help="Use data augmentation")
    
    args = vars(ap.parse_args())

    return args


def createOutputFolder(folder):
    # priprava prostoru pro ulozeni vystupu trenovani

    # vytvoreni slozky v pripade neexistence
    if not os.path.exists(folder):
        os.makedirs(folder)

    # vycisteni slozky
    for filename in os.listdir(folder):
        filePath = os.path.join(folder, filename)
        try:
            if os.path.isfile(filePath) or os.path.islink(filePath):
                os.unlink(filePath)
            elif os.path.isdir(filePath):
                shutil.rmtree(filePath)
        except Exception as e:
            print("Failed to clean output folder.")

    # vytvoreni slozky pro modely
    os.makedirs(folder + "/models")

    # vytvoreni slozky pro obrazky
    os.makedirs(folder + "/img")


def test(useBaseline, batchSize, pathToModel, useRectification, testImages, testLabels):
    
    print("Test - model: " + str(pathToModel))
    
    net = NetworkArchitecture(useBaseline, useRectification, imgSize=(imgWidth, imgHeight)).float().to(device)
    net.loadModel(pathToModel)
    net = net.float().to(device)
    
    indices = list(range(len(testImages)))
    batches = [indices[i * batchSize:(i + 1) * batchSize] for i in range((len(indices) + batchSize - 1) // batchSize )]

    criterion = nn.CrossEntropyLoss().to(device) if useBaseline else nn.CrossEntropyLoss(ignore_index=35).to(device)

    correctChars = 0
    result = [0,0,0,0,0,0,0,0,0]
    runningLoss = 0.0

    for batch in batches:
        imageBatch = np.zeros((len(batch), 3, imgHeight, imgWidth))
        labelBatch = np.zeros((len(batch), 8 if useBaseline else 10))

        for i, id in enumerate(batch):
            img = cv2.resize(testImages[id], dsize=(imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)
            imageBatch[i] = np.moveaxis(img,2,0)
            labelBatch[i] = Coder.encode(testLabels[id], useBaseline)

        imageBatch = torch.tensor(imageBatch).float().to(device)
        labelBatch = torch.tensor(labelBatch).float().to(device)

        _, _, out = net(imageBatch, labelBatch, isTrain=False)

        if useBaseline:
            (out1, out2, out3, out4, out5, out6, out7, out8) = out
            loss1 = criterion(out1, labelBatch[:,0].long())
            loss2 = criterion(out2, labelBatch[:,1].long())
            loss3 = criterion(out3, labelBatch[:,2].long())
            loss4 = criterion(out4, labelBatch[:,3].long())
            loss5 = criterion(out5, labelBatch[:,4].long())
            loss6 = criterion(out6, labelBatch[:,5].long())
            loss7 = criterion(out7, labelBatch[:,6].long())
            loss8 = criterion(out8, labelBatch[:,7].long())
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
        else:
            labelBatch = labelBatch[...,1:].long()
            loss = criterion(out.view(-1, out.shape[-1]), labelBatch.contiguous().view(-1))

        runningLoss += loss.item()

        for i, id in enumerate(batch):

            if useBaseline:
                tmp = np.zeros((8,36))
                tmp[0] = out1[i].detach().cpu().numpy()
                tmp[1] = out2[i].detach().cpu().numpy()
                tmp[2] = out3[i].detach().cpu().numpy()
                tmp[3] = out4[i].detach().cpu().numpy()
                tmp[4] = out5[i].detach().cpu().numpy()
                tmp[5] = out6[i].detach().cpu().numpy()
                tmp[6] = out7[i].detach().cpu().numpy()
                tmp[7] = out8[i].detach().cpu().numpy()
                value = Coder.decode(tmp, True)
            else:
                value = Coder.decode(out[i].detach().cpu().numpy(), False)

            for i in range(len(value)):
                if i > len(testLabels[id]) - 1:
                    break
                if value[i] == testLabels[id][i]:
                    correctChars += 1

            if value == testLabels[id]:
                result[0] += 1
            else:
                distance = Levenshtein.distance(value, testLabels[id])
                if distance > 8:
                    distance = 8
                result[distance] += 1

    print("Loss:", runningLoss / len(batches))
    print("Correct chars:", correctChars, "/", len(''.join(testLabels)))
    print("Levenshtein:", result)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    args = parseArgs(sys.argv)

    batchSize = int(args["batchSize"])
    runTest = args["test"]
    useRectification = args["rectification"]
    pathToModel = args["model"]
    epochs = 1
    outputFolder = args["outputFolder"]
    useBaseline = args["baseline"]
    useDataAugment = args["augmentation"]

    imgWidth = 200 if useBaseline else 100
    imgHeight = 40 if useBaseline else 32

    if runTest:
        if pathToModel is None:
            print("Path to model needs to be specified when running test.")
            exit(1)
    else:
        if epochs is None:
            print("Number of epochs needs to be specified to run training.")
            exit(1)
        elif outputFolder is None:
            print("Path to output folder needs to be specified to run training.")
            exit(1)

        epochs = int(args["epochs"])
        createOutputFolder(outputFolder)
        sys.stdout = open(str(outputFolder) + "/output.txt", "w")

    input = InputParser()

    trainImages = input.getImages("train")
    trainLabels = input.getLabels("train", useBaseline)
    testImages = input.getImages("test")
    testLabels = input.getLabels("test", useBaseline)
    validationImages = input.getImages("validation")
    validationLabels = input.getLabels("validation", useBaseline)

    if runTest:
        test(useBaseline, batchSize, pathToModel, useRectification, testImages, testLabels)
        exit()

    indices = list(range(len(trainImages)))
    validationIndices = list(range(len(validationImages)))
    outputLoss = np.zeros((epochs, 2))

    net = NetworkArchitecture(useBaseline, useRectification, imgSize=(imgWidth, imgHeight)).float().to(device)
    criterion = nn.CrossEntropyLoss() if useBaseline else nn.CrossEntropyLoss(ignore_index=35)

    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    
    originalImg = None
    rectificatedImg = None
    
    for epoch in range(epochs):

        runningLoss = 0.0
        validationRunningLoss = 0.0

        # -------------------------- TRAINING --------------------------

        random.shuffle(indices)
        batches = [indices[i * batchSize:(i + 1) * batchSize] for i in range((len(indices) + batchSize - 1) // batchSize )]
        
        for batch in batches:
            imageBatch = np.zeros((len(batch), 3, imgHeight, imgWidth))
            labelBatch = np.zeros((len(batch), 8 if useBaseline else 10))

            for i, id in enumerate(batch):
                img = cv2.resize(trainImages[id], dsize=(imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)
                if useDataAugment:
                    M = cv2.getRotationMatrix2D((imgWidth // 2, imgHeight // 2), random.randint(-10, 10), 1.0)
                    img = cv2.warpAffine(img, M, (imgWidth, imgHeight))
                imageBatch[i] = np.moveaxis(img,2,0)
                labelBatch[i] = Coder.encode(trainLabels[id], useBaseline)

            imageBatch = torch.tensor(imageBatch).float().to(device)
            labelBatch = torch.tensor(labelBatch).long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            originalImg, rectificatedImg, out = net(imageBatch, labelBatch)

            if useBaseline:
                (out1, out2, out3, out4, out5, out6, out7, out8) = out
                loss1 = criterion(out1, labelBatch[:,0].long())
                loss2 = criterion(out2, labelBatch[:,1].long())
                loss3 = criterion(out3, labelBatch[:,2].long())
                loss4 = criterion(out4, labelBatch[:,3].long())
                loss5 = criterion(out5, labelBatch[:,4].long())
                loss6 = criterion(out6, labelBatch[:,5].long())
                loss7 = criterion(out7, labelBatch[:,6].long())
                loss8 = criterion(out8, labelBatch[:,7].long())
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            else:
                labelBatch = labelBatch[...,1:]
                loss = criterion(out.view(-1, out.shape[-1]), labelBatch.contiguous().view(-1))
            
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            del imageBatch, labelBatch

        torch.save(net.state_dict(), str(outputFolder) + "/models/model_" + str(epoch))
        
        if useRectification:
            rectificatedImg = np.moveaxis(rectificatedImg.cpu().detach().numpy()[0],0,2)
            originalImg = np.moveaxis(originalImg.cpu().detach().numpy()[0],0,2)
            cv2.imwrite(str(outputFolder) + "/img/" + str(epoch) + "orig.png", cv2.cvtColor(originalImg, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(outputFolder) + "/img/" + str(epoch) + "test.png", cv2.cvtColor(rectificatedImg, cv2.COLOR_RGB2BGR))

        # ------------------------- VALIDATION -------------------------

        correctChars = 0
        result = [0,0,0,0,0,0,0,0,0]

        validationBatches = [validationIndices[i * batchSize:(i + 1) * batchSize] for i in range((len(validationImages) + batchSize - 1) // batchSize )]

        for batch in validationBatches:
            imageBatch = np.zeros((len(batch), 3, imgHeight, imgWidth))
            labelBatch = np.zeros((len(batch), 8 if useBaseline else 10))

            for i, id in enumerate(batch):
                img = cv2.resize(validationImages[id], dsize=(imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)
                imageBatch[i] = np.moveaxis(img,2,0)
                labelBatch[i] = Coder.encode(validationLabels[id], useBaseline)

            imageBatch = torch.tensor(imageBatch).float().to(device)
            labelBatch = torch.tensor(labelBatch).long().to(device)

            _, _, out = net(imageBatch, labelBatch)

            if useBaseline:
                (out1, out2, out3, out4, out5, out6, out7, out8) = out
                loss1 = criterion(out1, labelBatch[:,0].long())
                loss2 = criterion(out2, labelBatch[:,1].long())
                loss3 = criterion(out3, labelBatch[:,2].long())
                loss4 = criterion(out4, labelBatch[:,3].long())
                loss5 = criterion(out5, labelBatch[:,4].long())
                loss6 = criterion(out6, labelBatch[:,5].long())
                loss7 = criterion(out7, labelBatch[:,6].long())
                loss8 = criterion(out8, labelBatch[:,7].long())
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            else:
                labelBatch = labelBatch[...,1:]
                loss = criterion(out.view(-1, out.shape[-1]), labelBatch.contiguous().view(-1))

            validationRunningLoss += loss.item()

            for i, id in enumerate(batch):
                
                if useBaseline:
                    tmp = np.zeros((8,36))
                    tmp[0] = out1[i].detach().cpu().numpy()
                    tmp[1] = out2[i].detach().cpu().numpy()
                    tmp[2] = out3[i].detach().cpu().numpy()
                    tmp[3] = out4[i].detach().cpu().numpy()
                    tmp[4] = out5[i].detach().cpu().numpy()
                    tmp[5] = out6[i].detach().cpu().numpy()
                    tmp[6] = out7[i].detach().cpu().numpy()
                    tmp[7] = out8[i].detach().cpu().numpy()
                    value = Coder.decode(tmp, True)
                else:
                    value = Coder.decode(out[i].detach().cpu().numpy(), False)

                for i in range(len(value)):
                    if i > len(validationLabels[id]) - 1:
                        break
                    if value[i] == validationLabels[id][i]:
                        correctChars += 1

                if value == validationLabels[id]:
                    result[0] += 1
                else:
                    distance = Levenshtein.distance(value, validationLabels[id])
                    if distance > 8:
                        distance = 8
                    result[distance] += 1

            del imageBatch, labelBatch

        # --------------------------- OUTPUT ---------------------------

        print("----------------------------------------------------")
        print("Epoch:", epoch + 1)
        print("Validation loss:", validationRunningLoss / len(validationBatches))
        print("Training loss:", runningLoss / len(batches))
        print("Correct chars:", correctChars, "/", len(''.join(validationLabels)))
        print("Levenshtein:", result)

        outputLoss[epoch, 0] = runningLoss / len(batches)
        outputLoss[epoch, 1] = validationRunningLoss / len(validationBatches)

        with open(str(outputFolder) + "/lossRect.npy", 'wb') as f:
            np.save(f, outputLoss)
    
    print('Finished Training')

    if not runTest:
        sys.stdout.close()

    exit()