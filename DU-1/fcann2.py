import numpy as np
import torch
import data
import matplotlib.pyplot as plt

def fcann2_train(X, Y_):
    C = np.max(Y_) + 1
    K = 5
    param_delta = 0.01
    param_niter = 10000
    param_lambda = 0.001
    N, D = X.shape

    w1 = np.random.randn(D, K) 
    b1 = np.random.randn(K)

    w2 = np.random.randn(K, C)
    b2 = np.random.randn(C)

    for i in range(param_niter):
        # unaprijed
        S1 = np.dot(X, w1) + b1
        H = np.maximum(S1, 0)
        S2 = np.dot(H, w2) + b2
        expscores = np.exp(S2)
        sumexp = np.sum(expscores, axis=1)
        probs = np.transpose(np.transpose(expscores) / (sumexp))
        
        logprobs = np.log(probs)

        loss = -np.sum(logprobs[np.arange(X.shape[0]), Y_]) / X.shape[0] 
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # nazad
        y_vec = np.zeros((N, C))
        y_vec[np.arange(N), Y_] = 1

        dL_ds2 = probs - y_vec
        
        grad_w2 = np.dot(np.transpose(H), dL_ds2) / N
        grad_b2 = np.sum(dL_ds2, axis=0) / N

        dL_dh = np.dot(dL_ds2, np.transpose(w2))
        dL_ds1 = dL_dh * (S1 > 0)

        grad_w1 = np.dot(np.transpose(X), dL_ds1) / N
        grad_b1 = np.sum(dL_ds1, axis=0) / N

        w1 += -param_delta * grad_w1
        b1 += -param_delta * grad_b1

        w2 += -param_delta * grad_w2
        b2 += -param_delta * grad_b2

    return w1, b1, w2, b2

def fcann2_test():
    pass

def fcann2_classify(X, w1, b1, w2, b2):
    S1 = np.dot(X, w1) + b1
    H = np.maximum(S1, 0)
    S2 = np.dot(H, w2) + b2
    expscores = np.exp(S2)
    sumexp = np.sum(expscores, axis=1)
    probs = np.transpose(np.transpose(expscores) / (sumexp))
    return np.argmax(probs, axis=1)

def fcann2_decfun(w1, b1, w2, b2):
    def classify(X):
        return fcann2_classify(X, w1, b1, w2, b2)
    return classify

if __name__ == "__main__":
    # load the data
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    # train the model
    w1, b1, w2, b2 = fcann2_train(X, Y_)
    Y = fcann2_classify(X, w1, b1, w2, b2)

    # print the accuracy
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print("Accuracy: {}".format(accuracy))
    print("Recall: {}".format(recall))
    print("Precision: {}".format(precision))

    # visualize the decision surface
    data.graph_surface(fcann2_decfun(w1, b1, w2, b2), [np.min(X, axis=0), np.max(X, axis=0)])
    data.graph_data(X, Y_, Y)
    plt.show()