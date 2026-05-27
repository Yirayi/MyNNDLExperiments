import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np

def one_hot(labels, num_classes):
    labels_one_hot = np.zeros((len(labels),num_classes))
    labels_one_hot[np.arange(len(labels)), labels] = 1.0
    return labels_one_hot


def Get_Data():
    (train_data, train_label), (test_data, test_label) = cifar10.load_data()

    # 归一化
    x_data = train_data.astype('float32') / 255.
    y_data = test_data.astype('float32') / 255.  # ← 注意这里应该是 test_data

    num_classes = 10

    # 训练标签
    train_label = np.squeeze(train_label.astype('int32'))
    x_label = one_hot(train_label, num_classes)  # shape: (N, 10)

    # 测试标签：同样转 one-hot
    test_label = np.squeeze(test_label.astype('int32'))
    y_label = one_hot(test_label, num_classes)  # shape: (N, 10)

    return x_data, y_data, x_label, y_label

if __name__ == '__main__':
    print('GPU:', tf.config.list_physical_devices('GPU'))
    x_data, y_data, x_label, y_label = Get_Data()
    print(f"训练集数据:{x_data.shape},单个数据{x_data[0]}")
    print(f"训练集标签:{x_label.shape},单个数据{x_label[0]}")
    print(f"测试集数据:{y_data.shape},单个数据{y_data[0]}")
    '''
    训练集数据:(50000, 32, 32, 3)
    训练集标签:(50000, 10),单个数据[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    
    测试集数据:(10000, 32, 32, 3)
    '''