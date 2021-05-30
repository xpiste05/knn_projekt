import numpy as np

def encodeBaseline(label):

    output = np.zeros(8)

    for i, char in enumerate(label):
        if char.isdigit():
            output[i] = ord(char) - 48
        elif char == "#":
            output[i] = 24
        else:
            output[i] = ord(char) - 55

    return output


def decodeBaseline(encodedLabel):

    output = ""

    for encodedChar in encodedLabel:
        index = np.where(encodedChar == encodedChar.max())[0][0]
            
        if index == 24:
            output += "#"
        elif index < 10:
            output += chr(index + 48)
        else:
            output += chr(index + 55)

    return output


class Coder():
    def encode(label, useBaseline):

        if useBaseline:
            return encodeBaseline(label)

        output = np.zeros(10)
        output[0] = 35
        output[len(label):] = 36

        for i, char in enumerate(label):
            if char.isdigit():
                output[i + 1] = ord(char) - 48
            elif ord(char) < ord('O'):
                output[i + 1] = ord(char) - 55
            else:
                output[i + 1] = ord(char) - 56

        return output

    def decode(encodedLabel, useBaseline):

        if useBaseline:
            return decodeBaseline(encodedLabel)

        output = ""

        for encodedChar in encodedLabel:
            index = np.where(encodedChar == encodedChar.max())[0][0]
            
            if index == 35 or index == 36:
                continue
            elif index < 10:
                output += chr(index + 48)
            elif index < 24:
                output += chr(index + 55)
            else:
                output += chr(index + 56)

        return output
        