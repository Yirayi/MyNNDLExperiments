import scipy.io
import numpy as np
if __name__ == '__main__':
    for i in range(1,4):
        datastr = '../datasetSVM/dataset_'+str(i)+'.mat'
        print(datastr)
        data = scipy.io.loadmat(datastr)
        print(data.keys())
        for key, value in data.items():
            if not key.startswith('__'):  # 过滤掉元信息
                print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
        print("\n\n")

    datastr = '../spamTrain.mat'
    print(datastr)
    data = scipy.io.loadmat(datastr)
    print(data.keys())
    for key, value in data.items():
        if not key.startswith('__'):  # 过滤掉元信息
            print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
    print("\n\n")

    datastr = '../spamTest.mat'
    print(datastr)
    data = scipy.io.loadmat(datastr)
    print(data.keys())
    for key, value in data.items():
        if not key.startswith('__'):  # 过滤掉元信息
            print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
    print("\n\n")
