import os

class InputParser():
    def __init__(self):

        dataset_dir = "dataset"
        train_values_file = os.path.join(dataset_dir, "dataset.txt")

        with open(train_values_file, 'r', encoding="utf-8") as f:
            lines = [line.strip('\r\n') for line in f.readlines()]

        self.train_image_path_list = []
        self.train_value_list = []
        self.test_image_path_list = []
        self.test_value_list = []

        for i, line in enumerate(lines):
            split = line.split(';')
            img_path = split[0]
            value = split[1]
            train = split[2]

            if train == "1":
                self.train_image_path_list.append(os.path.join(dataset_dir, img_path))
                self.train_value_list.append(value)
            else:
                self.test_image_path_list.append(os.path.join(dataset_dir, img_path))
                self.test_value_list.append(value)

        size = len(self.train_image_path_list) - (len(self.train_image_path_list) // 10)
        self.validation_image_path_list = self.train_image_path_list[size:]
        self.train_image_path_list = self.train_image_path_list[:size]
        self.validation_value_list = self.train_value_list[size:]
        self.train_value_list = self.train_value_list[:size]

        print(len(self.train_image_path_list))
        print(len(self.test_image_path_list))


    def getPathsToImages(self, setType):

        if setType == "train":
            return self.train_image_path_list
        elif setType == "validation":
            return self.validation_image_path_list
        else:
            return self.test_image_path_list


    def getLabels(self, setType):

        labels = []

        if setType == "train":
            labels = self.train_value_list
        elif setType == "validation":
            labels = self.validation_value_list
        else:
            labels = self.test_value_list

        for i, label in enumerate(labels):
            size = len(label)
            if size > 8:
                print(label)
            if size < 8:
                labels[i] = label[:size - 4] + "#" * (8 - size) + label[size - 4:]

        return labels
