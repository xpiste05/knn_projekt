import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from PIL import Image
from network import Net 
from inputParser import InputParser

def encodeLabel(label):
    output = np.zeros((8, 36))

    for i, char in enumerate(label):
        if char.isdigit():
            output[i, ord(char) - 48] = 1
        elif char == "#":
            output[i, 24] = 1
        else:
            output[i, ord(char) - 55] = 1

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

    for epoch in range(2):
        random.shuffle(indices)
        final = [indices[i * batch_size:(i + 1) * batch_size] for i in range((len(indices) + batch_size - 1) // batch_size )]

        for batch in final:
            imageBatch = np.zeros((len(batch), 3, 40, 200))
            labelBatch = np.zeros((len(batch), 8, 36))

            for i, id in enumerate(batch):
                img = Image.open(train_images[id], mode='r')
                img = img.resize((200, 40))
                imageBatch[i] = np.asarray(img)
                labelBatch[i] = encodeLabel(train_labels[id])
                exit()

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    for epoch in range(2):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # 1. prevest obrazek na 200x40x3
            # 2. inputs je 4D vektor: batch_size, channels, height, width
            # 3. out1 je 2D vektor: batch_size, 36(pocet znaku + mrizka pro volnou pozici)
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out1, out2, out3, out4, out5, out6, out7, out8 = net(inputs)
            loss1 = criterion(out1, label1)
            loss2 = criterion(out2, label2)
            loss3 = criterion(out3, label3)
            loss4 = criterion(out4, label4)
            loss5 = criterion(out5, label5)
            loss6 = criterion(out6, label6)
            loss7 = criterion(out7, label7)
            loss8 = criterion(out8, label8)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')