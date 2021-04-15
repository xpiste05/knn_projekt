import torch.nn as nn
import torch.optim as optim
import random
import torch
import numpy as np
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

if __name__ == "__main__":

    batch_size = 64
    input = InputParser()
    train_images = input.getPathsToImages(train=True)
    train_labels = input.getLabels(train=True)
    indices = list(range(len(train_images)))

    net = Net().float()
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()
    criterion4 = nn.CrossEntropyLoss()
    criterion5 = nn.CrossEntropyLoss()
    criterion6 = nn.CrossEntropyLoss()
    criterion7 = nn.CrossEntropyLoss()
    criterion8 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        running_loss = 0.0
        random.shuffle(indices)
        final = [indices[i * batch_size:(i + 1) * batch_size] for i in range((len(indices) + batch_size - 1) // batch_size )]

        for batch in final:
            imageBatch = np.zeros((len(batch), 3, 40, 200))
            labelBatch = np.zeros((len(batch), 8))

            for i, id in enumerate(batch):
                img = Image.open(train_images[id], mode='r')
                img = img.resize((200, 40))
                imageBatch[i] = np.moveaxis(np.asarray(img),2,0)
                labelBatch[i] = encodeLabel(train_labels[id])

            imageBatch = torch.tensor(imageBatch).float()
            labelBatch = torch.tensor(labelBatch).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out1, out2, out3, out4, out5, out6, out7, out8 = net(imageBatch)
            loss1 = criterion(out1, labelBatch[:,0].long())
            loss2 = criterion2(out2, labelBatch[:,1].long())
            loss3 = criterion3(out3, labelBatch[:,2].long())
            loss4 = criterion4(out4, labelBatch[:,3].long())
            loss5 = criterion5(out5, labelBatch[:,4].long())
            loss6 = criterion6(out6, labelBatch[:,5].long())
            loss7 = criterion7(out7, labelBatch[:,6].long())
            loss8 = criterion8(out8, labelBatch[:,7].long())
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
#            if i % 2000 == 1999:    # print every 2000 mini-batches
#                print('[%d, %5d] loss: %.3f' %
#                    (epoch + 1, i + 1, running_loss / 2000))
#                running_loss = 0.0
        print(running_loss / len(final), epoch)
        for ind in [0,2,14]:
            pom = np.zeros((1,3,40,200))
            img = Image.open(train_images[ind], mode='r')
            img = img.resize((200, 40))
            img = np.moveaxis(np.asarray(img),2,0)
            pom[0] = img
            pom = torch.tensor(pom).float()
            o1, o2, o3, o4, o5, o6, o7, o8 = net(pom)
            pom2 = np.zeros((8,36))
            pom2[0] = o1[0].detach().numpy()
            pom2[1] = o2[0].detach().numpy()
            pom2[2] = o3[0].detach().numpy()
            pom2[3] = o4[0].detach().numpy()
            pom2[4] = o5[0].detach().numpy()
            pom2[5] = o6[0].detach().numpy()
            pom2[6] = o7[0].detach().numpy()
            pom2[7] = o8[0].detach().numpy()
            print(decodeLabel(pom2), train_labels[ind])

    print('Finished Training')
