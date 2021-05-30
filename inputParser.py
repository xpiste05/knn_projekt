import os
import numpy as np

from PIL import Image

class InputParser():
    def __init__(self):

        if not os.path.isfile("TrainDataset.npz") or not os.path.isfile("TestDataset.npz"):

            datasetDir = "dataset"
            valuesFile = os.path.join(datasetDir, "dataset.txt")

            with open(valuesFile, 'r', encoding="utf-8") as f:
                lines = [line.strip('\r\n') for line in f.readlines()]

            trainImagePathList = []
            trainValueList = []
            testImagePathList = []
            testValueList = []

            for i, line in enumerate(lines):
                split = line.split(';')
                img_path = split[0]
                value = split[1]
                train = split[2]

                if train == "1":
                    trainImagePathList.append(os.path.join(datasetDir, img_path))
                    trainValueList.append(value)
                else:
                    testImagePathList.append(os.path.join(datasetDir, img_path))
                    testValueList.append(value)

            self.createDatasetFile(trainImagePathList, trainValueList, "TrainDataset.npz")
            self.createDatasetFile(testImagePathList, testValueList, "TestDataset.npz")
        
        
        self.loadDataset(train = True)
        self.loadDataset(train = False)

        # TODO: omezeni
        # self.trainImages = self.trainImages[::20]
        # self.trainLabels = self.trainLabels[::20]           


    def loadImg(self, path):
        with Image.open(path) as img:
            return np.asarray(img)

    def createDatasetFile(self, imagePaths, labels, datasetName):

        dataset = np.empty([len(imagePaths),2], dtype=object)

        for i in range(len(imagePaths)):
            dataset[i][0] = self.loadImg(imagePaths[i])
            dataset[i][1] = labels[i]
            if i%5000 == 0:
                print(i, "/", len(imagePaths))

        np.savez(datasetName, data=dataset)

    def loadDataset(self, train):
        
        datasetName = "TrainDataset.npz" if train else "TestDataset.npz"

        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True)

        dataset = np.load(datasetName)
        data = dataset['data']

        if train:
            self.trainImages = data[..., 0]
            self.trainLabels = data[..., 1]
        else:
            self.testImages = data[..., 0]
            self.testLabels = data[..., 1]

    def getImages(self, setType):

        N = 10
        images = []

        if setType == "train":
            images = self.trainImages
            images = np.delete(images, np.arange(0, images.size, N))
        elif setType == "validation":
            images = self.trainImages[::N]
        else:
            images = self.testImages

        print(setType, len(images))
        return images

    def getLabels(self, setType, useBaseline):

        N = 10
        labels = []

        if setType == "train":
            labels = self.trainLabels
            labels = np.delete(labels, np.arange(0, labels.size, N))
        elif setType == "validation":
            labels = self.trainLabels[::N]
        else:
            labels = self.testLabels

        if useBaseline:
            for i, label in enumerate(labels):
                size = len(label)
                if size < 8:
                    labels[i] = label[:size - 4] + "#" * (8 - size) + label[size - 4:]

        return labels
