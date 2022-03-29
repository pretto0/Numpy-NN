import numpy as np
import ReadMNIST
import matplotlib.pyplot as plt


# def tanh(x):
#     s = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
#     return s
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

def Exponential_Decay(lr, i, beta=0.96):
    return lr * pow(beta, i)

def Relu(x):
    return np.maximum(0,x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def initParameters(layer_sizei, layer_size1, layer_sizeO):

    W_1 = np.random.randn(layer_size1,layer_sizei)*0.01
    b_1 = np.zeros((layer_size1,1))
    W_2 = np.random.randn(layer_sizeO,layer_size1)*0.01
    #*0.01是为了保证值落在激活函数斜率绝对值较大的部分，加快学习速度
    b_2 = np.zeros((layer_sizeO,1))
    print('W_1.shape:' + str(W_1.shape))
    print('W_2.shape:' + str(W_2.shape))
    print('b_1.shape:' + str(b_1.shape))
    print('b_2.shape:' + str(b_2.shape))
    return W_1,W_2,b_1,b_2


def acc_caculate(y,yhat):
    """计算准确率"""
    yhat = yhat.T
    yhat = (yhat == yhat.max(axis=1, keepdims=1)).astype(float)
    yhat = yhat.T
    # print(yhat.shape)
    acc = np.sum(y * yhat) / float(len(yhat.T))
    return acc,yhat


def optimizeGD(W_1,W_2,b_1,b_2,lr,grads):
    """梯度下降方法"""
    W_1 = W_1 - lr * grads['dw1']
    W_2 = W_2 - lr * grads['dw2']
    b_1 = b_1 - lr * grads['db1']
    b_2 = b_2 - lr * grads['db2']

    return W_1,W_2,b_1,b_2


def back_propagation(x, y, yhat, a_1, W_1, W_2, N, lam):
    assert (y.shape == yhat.shape)
    """后向传播"""
    dZ_2 = (1 / N) * (yhat - y)
    # print('dZ_2.shape:'+str(dZ_2.shape))
    dW_2 = np.dot(dZ_2, a_1.T) + lam * W_2#(10*16)
    db_2 = dZ_2.sum(axis=1, keepdims=True)
    dZ_1 = np.dot(W_2.T, dZ_2) * (a_1 > 0)#(16*60000)
    dW_1 = np.dot(dZ_1, x.T) + lam * W_1#(16*60000)*(60000*784)
    db_1 = dZ_1.sum(axis=1, keepdims=True)
    grads = {'dw2': dW_2,
             'db2': db_2,
             'dw1': dW_1,
             'db1': db_1}
    return grads


def forward_propagation(W_1, W_2, b_1, b_2, data):
    z_1 = np.dot(W_1, data) + b_1 #(16,1)
    #broadcasting python的自动补全
    a_1 = Relu(z_1)
    z_2 = np.dot(W_2, a_1) + b_2
    yhat = softmax(z_2)
    # print('z_1.shape and a_1.shape:' + str(z_1.shape))
    # print('z_2.shape and yhat.shape:' + str(z_2.shape))
    return yhat, a_1


def cross_entropy_error(yhat, y, batch_size, W_1, W_2, lam):
    assert(y.shape == yhat.shape)
    print('batchsize:'+str(batch_size))
    loss1 = (-1/batch_size)*np.sum((y*np.log(yhat) + (1-y)*np.log(1-yhat)))
    loss2 = 1/2 * lam * (np.sum(np.square(W_2)) + np.sum(np.square(W_1)))
    print('loss1：'+str(loss1)+'   ''loss2：'+str(loss2))
    return loss1 + loss2


def NN_model(train_data, train_labels,lrate,layer_size1 = 20,lam = 0.1):

    layer_sizei, layer_sizeO = 784, train_labels.shape[0]
    W_1, W_2, b_1, b_2 = initParameters(layer_sizei, layer_size1, layer_sizeO)


    # 绘制动态折线图
    plt.figure(figsize=(20, 10), dpi=100)
    xs = []
    ys = []
    ls = []

    for i in range(5001):
        # 前向计算
        x = np.random.randint(train_data.shape[1])
        SGDdata = train_data[:,x].reshape(784,1)
        SGDlabels = train_labels[:,x].reshape(10,1)

        yhat, a_1 = forward_propagation(W_1, W_2, b_1, b_2, SGDdata)#(10,60000),(16,60000)
        ytest,_ = forward_propagation(W_1, W_2, b_1, b_2, train_data)
        if i % 50 == 0:
            loss = cross_entropy_error(yhat, SGDlabels,SGDdata.shape[1],W_1,W_2,lam)
            print('cur_iters:'+str(i)+'    '+'loss:'+str(loss))
            acc, _ = acc_caculate(train_labels, ytest)
            print('cur_iters:'+str(i)+'    '+'acc:'+str(acc))

            plt.subplot(1, 2, 1)
            try:
                train_loss_lines.remove(train_loss_lines[0])  # 移除上一步曲线
            except Exception:
                pass

            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title("loss",fontdict={'size': 20})
            plt.xlabel("iters",fontdict={'size': 16})
            plt.ylabel("loss",fontdict={'size': 16})
            train_loss_lines = plt.plot(xs, ls, 'r', lw=1)


            plt.subplot(1, 2, 2)
            try:
                train_acc_lines.remove(train_acc_lines[0])  # 移除上一步曲线
            except Exception:
                pass
            plt.xlabel("iters", fontdict={'size': 16})
            plt.ylabel("acc", fontdict={'size': 16})
            plt.title("acc_rate", fontdict={'size': 20})
            plt.grid(True, linestyle='--', alpha=0.5)

            ys.append(acc)
            ls.append(loss)
            xs.append(i)
            train_acc_lines = plt.plot(xs, ys, 'r', lw=1)

            plt.pause(0.1)

        # 后向传播计算梯度
        grads = back_propagation(SGDdata, SGDlabels, yhat, a_1, W_1,W_2, SGDdata.shape[1],lam)
        # 更新参数
        W_1, W_2, b_1, b_2 = optimizeGD(W_1, W_2, b_1, b_2, Exponential_Decay(lrate, i//50, beta=0.96), grads)

    plt.savefig('acc_and_loss_rate.png')



    model_parameters = {'w2': W_2, 'b2': b_2, 'w1':W_1, 'b1': b_1}
    return model_parameters


def test(test_data,model_parameters,test_labels):
    W_1,W_2,b_1,b_2 = model_parameters['w1'],model_parameters['w2'],model_parameters['b1'],model_parameters['b2']
    yhat,_ = forward_propagation(W_1, W_2, b_1, b_2, test_data)
    acc,_ = acc_caculate(test_labels, yhat)
    print('test_acc='+str(acc))


if __name__ == '__main__':
    np.random.seed(1)

    train_data, train_labels, test_data, test_labels = ReadMNIST.readMNIST()
    print('train_data.shape:'+str(train_data.shape))
    print('train_labels.shape:' + str(train_labels.shape))
    print('test_data.shape:' + str(test_data.shape))
    print('test_labels.shape:' + str(test_labels.shape))
    model_parameters = NN_model(train_data, train_labels,0.01, 20, 0.1)

    test(test_data,model_parameters,test_labels)

    np.savetxt("W_1.csv", model_parameters['w1'], delimiter=",")
    np.savetxt("W_2.csv", model_parameters['w2'], delimiter=",")
    np.savetxt("b_1.csv", model_parameters['b1'], delimiter=",")
    np.savetxt("b_2.csv", model_parameters['b2'], delimiter=",")




