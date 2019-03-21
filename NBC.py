
import pandas as pd
import csv
import random
import math
import sys

''' Preprocess reads tsv file and  returns list of instances and class probabilities
'''
def Preprocess(filename):
    data = pd.read_table(filename, delim_whitespace=True, header=None)
    data[0] = data[0].map({'A': 1, 'B': 0})
    c1 = 0
    c2 = 0
    for i in range(len(data[0])):
        if data[0][i]== 1:
            c1+=1
        else:
            c2+=1
    pc1=c1/len(data[0])  #calculating class probabilities
    pc2=c2/len(data[0])
    dataset=data.values.tolist()
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset,pc1,pc2

"""split_data takes filename as input, reads the dataframe from it
and splits the dataframe in test data and training data"""
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

'''
Separate instances based on class label
'''
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]

        if (vector[0] not in separated):
            separated[vector[0]] = []
        separated[vector[0]].append(vector)
        #print('Separated',separated)
    return separated

'''
calculate mean for attribute
'''
def mean(numbers):
    return sum(numbers) / float(len(numbers))

'''
Compute variance for attribute
'''
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return variance


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[0]
    return summaries

'''
summarize mean ,variance values for each class 
'''
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)


    return summaries

'''Calculate gaussian density function'''
def calculateGaussianDensity(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    prob = (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    #print(prob)
    return prob

'''Calculate likelihood probability for attributes '''
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}

    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i+1]
            probabilities[classValue] *= calculateGaussianDensity(x, mean, stdev)

    return probabilities

'''Predict class labels of test dataset
 attributes using likelihood probabilities calculated above'''
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    # print (probabilities)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb  = probability
            bestLabel = classValue
    return bestLabel


def getPredictedlabel(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
        #print(result)
    return predictions

def getmissclassification(testSet, predictions):
    correct = 0
    incorrect=0
    for i in range(len(testSet)):
        if testSet[i][0] == predictions[i]:
            correct += 1
        else:
            incorrect+=1
    percent= (correct / float(len(testSet))) * 100.0
    return percent,incorrect
'''format output for writing to tsv file'''
def out(summaries,p1,p2,inc):
    class1 = []
    class2 = []
    out1=[]
    for key, values in summaries.items():
        if key == 1:
            for i in range(len(values)):
                for j in range(len(values[i])):
                    class1.append(values[i][j])
        elif key == 0:
            for i in range(len(values)):
                for j in range(len(values[i])):
                    class2.append(values[i][j])
    class1.append(p1)
    class2.append(p2)
    out1.append(inc)
    return class1,class2,out1

if __name__ == '__main__':
    file = sys.argv[1]  # input file path
    file1 = sys.argv[2]  # output file path
    splitRatio = 0.60
    dataset,pc1,pc2 = Preprocess(file)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    summaries = summarizeByClass(dataset)
    #print(summaries)
    predictions= getPredictedlabel(summaries, testSet)
    accuracy,inc = getmissclassification(testSet, predictions)
   # print('Accuracy: ', accuracy,"mis",inc)
    out1,out2,out3=out(summaries,pc1,pc2,inc)
    with open(file1, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(out1)
        tsv_writer.writerow(out2)
        tsv_writer.writerow(out3)



