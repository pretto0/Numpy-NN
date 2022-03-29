import numpy as np
import ReadMNIST
import train

if __name__ == '__main__':

    W_1 = np.loadtxt("./model/W_1.csv",delimiter=",")
    W_2 = np.loadtxt("./model/W_2.csv",delimiter=",")
    b_1 = np.loadtxt("./model/b_1.csv",delimiter=",").reshape(20,1)
    b_2 = np.loadtxt("./model/b_2.csv",delimiter=",").reshape(10,1)


    _, train_labels, test_data, test_labels = ReadMNIST.readMNIST()

    yhat, _ = train.forward_propagation(W_1, W_2, b_1, b_2, test_data)

    acc,_ = train.acc_caculate(test_labels, yhat)

    print('the acc for test_data is:'+str(acc))

