import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselineRecognitionNetwork import BaselineRecognitionNetwork
from .recognitionNetwork import RecognitionNetwork
from .rectificationTransformer import RectificationTransformer
from .localizationNetwork import LocalizationNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NetworkArchitecture(nn.Module):

    def __init__(self, useBaseline, useRectification, imgSize):
        super().__init__()

        self.useRectification = useRectification
        self.useBaseline = useBaseline
        self.imgWidth, self.imgHeight = imgSize

        if useBaseline:
            self.recognitionNet = BaselineRecognitionNetwork().float().to(device)
        else:
            self.recognitionNet = RecognitionNetwork().float().to(device)

        if useRectification:
            self.localizationNetwork = LocalizationNetwork(numOfControlPoints=10).float().to(device)
            self.rectificationTransformer = RectificationTransformer(imgSize, numOfControlPoints=10).float().to(device)

        print("Init ended.")


    def forward(self, img, labels, isTrain=True):

        rectificatedImg = None
        
        if self.useRectification:
            controlPoints = self.localizationNetwork(img)
            rectificatedImg = self.rectificationTransformer(img, controlPoints)

        recognitionInput = rectificatedImg if self.useRectification else img
        output = self.recognitionNet(recognitionInput) if self.useBaseline else self.recognitionNet(recognitionInput, labels, isTrain)

        return img, rectificatedImg, output


    def loadModel(self, path):
        device = torch.device('cpu')
        self.load_state_dict(torch.load(path, map_location=device))
        self.float().to(device)