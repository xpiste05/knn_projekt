import torch.nn as nn
import torch.optim as optim
import random
import torch
import numpy as np
import datetime
import Levenshtein
from multiprocessing import Pool
from PIL import Image
from network import Net 
from inputParser import InputParser

def encodeLabel2(label):
    output = np.zeros((8, 36))

    for i, char in enumerate(label):
        if char.isdigit():
            output[i, ord(char) - 48] = 1
        elif char == "#":
            output[i, 24] = 1
        else:
            output[i, ord(char) - 55] = 1

    return output

def encodeLabel(label):
    output = np.zeros(8)

    for i, char in enumerate(label):
        if char.isdigit():
            output[i] = ord(char) - 48
        elif char == "#":
            output[i] = 24
        else:
            output[i] = ord(char) - 55

    return output

def decodeLabel(encodedLabel):
    text = ""

    for encodedChar in encodedLabel:
        index = np.where(encodedChar == encodedChar.max())[0][0]

        if index == 24:
            text += "#"
        elif index < 10:
            text += chr(index + 48)
        else:
            text += chr(index + 55)

    return text

def loadImg(path):
    img = Image.open(path)
    img = img.resize((200,40))
    return np.moveaxis(np.asarray(img),2,0)

def loadModel(epoch):
    device = torch.device('cpu')
    model = Net()
    model.load_state_dict(torch.load("model_" + str(epoch), map_location=device))
    return model.float().cuda()

def test(batch_size, testImages, testLabels, epoch):
    net = loadModel(epoch) 
    indices = list(range(len(testImages)))
    batches = [indices[i * batch_size:(i + 1) * batch_size] for i in range((len(indices) + batch_size - 1) // batch_size )]

    correctChars = 0
    result = [0,0,0,0,0,0,0,0,0]

    for batch in batches:
        imageBatch = np.zeros((len(batch), 3, 40, 200))
        labelBatch = np.zeros((len(batch), 8))

        p = Pool()
        imageBatch = p.map(loadImg, [testImages[id] for id in batch])
        p.close()

        imageBatch = torch.tensor(imageBatch).float().cuda()

        out1, out2, out3, out4, out5, out6, out7, out8 = net(imageBatch)

        for i, id in enumerate(batch):
            tmp = np.zeros((8,36))
            tmp[0] = out1[i].detach().cpu().numpy()
            tmp[1] = out2[i].detach().cpu().numpy()
            tmp[2] = out3[i].detach().cpu().numpy()
            tmp[3] = out4[i].detach().cpu().numpy()
            tmp[4] = out5[i].detach().cpu().numpy()
            tmp[5] = out6[i].detach().cpu().numpy()
            tmp[6] = out7[i].detach().cpu().numpy()
            tmp[7] = out8[i].detach().cpu().numpy()
            value = decodeLabel(tmp)

            for i in range(len(value)):
                if value[i] == testLabels[id][i]:
                    correctChars += 1

            if value == testLabels[id]:
                result[0] += 1
            else:
                distance = Levenshtein.distance(value, testLabels[id])
                if distance > 8:
                    distance = 8
                result[distance] += 1

    print(correctChars, "z", len(testLabels) * 8)
    print(result)


if __name__ == "__main__":

    batch_size = 256
    epochs = 50

    input = InputParser()

    trainImages = input.getPathsToImages("train")
    trainLabels = input.getLabels("train")
    testImages = input.getPathsToImages("test")
    testLabels = input.getLabels("test")
    validationImages = input.getPathsToImages("validation")
    validationLabels = input.getLabels("validation")
    indices = list(range(len(trainImages)))
    validationIndices = list(range(len(validationImages)))
    outputLoss = np.zeros((epochs, 2))

    test(batch_size, testImages, testLabels, 4)
    exit()

    net = Net().float().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):

        running_loss = 0.0
        validation_running_loss = 0.0

        random.shuffle(indices)

        batches = [indices[i * batch_size:(i + 1) * batch_size] for i in range((len(indices) + batch_size - 1) // batch_size )]

        for batch in batches:
            imageBatch = np.zeros((len(batch), 3, 40, 200))
            labelBatch = np.zeros((len(batch), 8))

            p = Pool()
            imageBatch = p.map(loadImg, [trainImages[id] for id in batch])
            p.close()

            for i, id in enumerate(batch):
                labelBatch[i] = encodeLabel(trainLabels[id])

            imageBatch = torch.tensor(imageBatch).float().cuda()
            labelBatch = torch.tensor(labelBatch).float().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out1, out2, out3, out4, out5, out6, out7, out8 = net(imageBatch)
            loss1 = criterion(out1, labelBatch[:,0].long())
            loss2 = criterion(out2, labelBatch[:,1].long())
            loss3 = criterion(out3, labelBatch[:,2].long())
            loss4 = criterion(out4, labelBatch[:,3].long())
            loss5 = criterion(out5, labelBatch[:,4].long())
            loss6 = criterion(out6, labelBatch[:,5].long())
            loss7 = criterion(out7, labelBatch[:,6].long())
            loss8 = criterion(out8, labelBatch[:,7].long())
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
#                running_loss = 0.0

        torch.save(net.state_dict(), "model_" + str(epoch))

        validationBatches = [validationIndices[i * batch_size:(i + 1) * batch_size] for i in range((len(validationImages) + batch_size - 1) // batch_size )]

        for batch in validationBatches:
            imageBatch = np.zeros((len(batch), 3, 40, 200))
            labelBatch = np.zeros((len(batch), 8))

            p = Pool()
            imageBatch = p.map(loadImg, [validationImages[id] for id in batch])
            p.close()

            for i, id in enumerate(batch):
                labelBatch[i] = encodeLabel(validationLabels[id])

            imageBatch = torch.tensor(imageBatch).float().cuda()
            labelBatch = torch.tensor(labelBatch).float().cuda()

            out1, out2, out3, out4, out5, out6, out7, out8 = net(imageBatch)
            loss1 = criterion(out1, labelBatch[:,0].long())
            loss2 = criterion(out2, labelBatch[:,1].long())
            loss3 = criterion(out3, labelBatch[:,2].long())
            loss4 = criterion(out4, labelBatch[:,3].long())
            loss5 = criterion(out5, labelBatch[:,4].long())
            loss6 = criterion(out6, labelBatch[:,5].long())
            loss7 = criterion(out7, labelBatch[:,6].long())
            loss8 = criterion(out8, labelBatch[:,7].long())
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8

            validation_running_loss += loss.item()

        print("Validation loss:", validation_running_loss / len(validationBatches), epoch)
        print("Training loss:", running_loss / len(batches), epoch)
        outputLoss[epoch, 0] = running_loss / len(batches)
        outputLoss[epoch, 1] = validation_running_loss / len(validationBatches)

        with open('loss.npy', 'wb') as f:
            np.save(f, outputLoss)









        #for ind in [0,2,14]:
        #    pom = np.zeros((1,3,40,200))
        #    img = Image.open(trainImages[ind], mode='r')
        #    img = img.resize((200, 40))
        #    img = np.moveaxis(np.asarray(img),2,0)
        #    pom[0] = img
        #    pom = torch.tensor(pom).float().cuda()
        #    o1, o2, o3, o4, o5, o6, o7, o8 = net(pom)
        #    pom2 = np.zeros((8,36))
        #    pom2[0] = o1[0].detach().numpy()
        #    pom2[1] = o2[0].detach().numpy()
        #    pom2[2] = o3[0].detach().numpy()
        #    pom2[3] = o4[0].detach().numpy()
        #    pom2[4] = o5[0].detach().numpy()
        #    pom2[5] = o6[0].detach().numpy()
        #    pom2[6] = o7[0].detach().numpy()
        #    pom2[7] = o8[0].detach().numpy()
        #    print(decodeLabel(pom2), trainLabels[ind])
    
    print('Finished Training')
