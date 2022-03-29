import numpy as np

def readMNIST():
    # 训练集
    with open('./MNIST/train-images.idx3-ubyte') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        train_data = loaded[16:].reshape((60000, 784))

    with open('./MNIST/train-labels.idx1-ubyte') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        train_labels = loaded[8:].reshape((60000, 1))

    # 测试集
    with open('./MNIST/t10k-images.idx3-ubyte') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        test_data = loaded[16:].reshape((10000, 784))

    with open('./MNIST/t10k-labels.idx1-ubyte') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        test_labels = loaded[8:].reshape((10000, 1))


    return standardization(train_data/255.0).T,np.eye(10)[train_labels].reshape(60000,10).T,standardization(test_data/255.0).T,np.eye(10)[test_labels].reshape(10000,10).T



def standardization(data):
    mu = np.mean(data, axis=1).reshape(data.shape[0],1)
    sigma = np.std(data, axis=1).reshape(data.shape[0],1)
    return (data - mu) / sigma
#标准化操作


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = readMNIST()

    print(train_data.shape)
    print(train_data[1])
    # trainX_trans = train_data.reshape((train_data.shape[0], -1)).T
    # testX_trans = test_data.reshape((test_data.shape[0], -1)).T
