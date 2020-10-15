import os
import numpy as np
import random

def ReadFile(fileName):
    f = open(fileName, mode='r')
    return f.readlines()

def GetTrainingData(rawData):
    '''
    Return data include:
    training_data: 
        A 2-D numpy array which first dimension is training datas; 
        second dimension are the features of each data.
    training_label:
        A 1-D numpy array which store all labels. Index 0 infer the label of first data.
    '''
    data_list_str = []
    
    for data in rawData:
        data_list_str.append(data.split())
    A = np.array(data_list_str)
    A = A.astype('float32')

    training_data = []
    training_label = []
    x0 = []
    for a in A:
        training_data.append(a[0:10])
        training_label.append(a[10])
        x0.append([0.0])
    x0 = np.array(x0)
    training_data = np.array(training_data)
    training_label = np.array(training_label)
    # add x0 = 1 to every xn
    training_data = np.concatenate([x0, training_data], axis=1)
    return (training_data, training_label)

def checkSign(wx):
    if wx <= 0:
        return -1
    else:
        return 1

def start():
    rawData = ReadFile('../hw1_train.dat')
    (datas, labels) = GetTrainingData(rawData)
    datas = datas/4
    dataLen = labels.size

    update_times_list = []
    w0_list = []
    for repeatTime in range(1000):
        w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        correct_times = 0
        update_times = 0
        target_time = 5*dataLen
        while correct_times < target_time:
            checkIndex = random.randint(0, dataLen-1)
            wx = w.dot(datas[checkIndex])
            if checkSign(wx) * labels[checkIndex] < 0:
                w = w + labels[checkIndex]*datas[checkIndex]
                correct_times = 0
                update_times = update_times+1
                #print(update_times)
            else:
                correct_times = correct_times+1
        update_times_list.append(update_times)
        w0_list.append(w[0])
    update_times_list.sort()
    w0_list.sort()
    print(" The median # of updates before returns wPLA is: " + str(update_times_list[500]))
    print(" The median # of w0 is: " + str(w0_list[500]))




if __name__ == "__main__":
    start()