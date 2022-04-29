import numpy as np
from sklearn.ensemble import RandomForestClassifier

trainList = []
testList = []


class Domain:
    def __init__(self, _name: str, _label: bool):
        self.name = _name
        self.label = _label
        self.nameLen = len(self.name)

        charNum = dict({"letSum": 0, "numSum": 0, "dotSum": 0})
        for i in self.name:
            if i in "0123456789":
                charNum["numSum"] += 1
            elif i in "qwertyuiopasdfghjklzxcvbnm":
                charNum["letSum"] += 1
                if i in charNum.keys():
                    charNum[i] += 1
                else:
                    charNum[i] = 1
            else:
                charNum["dotSum"] += 1

        self.nameNumLen = charNum["numSum"]
        self.entropy = 0
        for i in charNum.keys():
            if len(i) < 2:
                p = charNum[i] / charNum["letSum"]
                self.entropy -= (p * np.log2(p))

        self.segmentation = charNum["dotSum"]

    def getData(self):
        return [
            self.nameLen, self.nameNumLen, (self.entropy), self.segmentation
        ]

    def getLabel(self):
        return self.label


def trainData(filename):
    with open(filename) as f:
        for line in f:
            tokens = line.split(",")
            name = tokens[0].strip()
            label = tokens[1].strip()
            trainList.append(Domain(name, label == "dga"))


def testData(filename, clf):
    with open(filename) as f:
        for line in f:
            testList.append(Domain(line.strip(), False))
    predictMatrix = []
    for item in testList:
        predictMatrix.append(item.getData())
    result = clf.predict(predictMatrix)
    for i in range(len(testList)):
        with open("result.txt", "a") as f:
            f.write(testList[i].name + ',' +
                    ("dga" if result[i] else "notdga") + '\n')


def main():
    trainData("train.txt")
    featureMatrix = []
    labelList = []
    for item in trainList:
        featureMatrix.append(item.getData())
        labelList.append(int(item.getLabel()))

    clf = RandomForestClassifier()
    clf.fit(featureMatrix, labelList)

    testData("test.txt", clf)


if __name__ == '__main__':
    main()
