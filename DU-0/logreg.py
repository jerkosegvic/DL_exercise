import numpy as np
import matplotlib.pyplot as plt
import random
import data

def logreg_train(X, Y_):
    c = np.max(Y_) + 1
    param_niter = 10000
    param_delta = 0.025

    W = np.random.randn(c, X.shape[1])
    b = np.zeros(c)

    for i in range(param_niter):
        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        #breakpoint()
        scores = np.dot(X, np.transpose(W)) + b    # N x C
        expscores = np.exp(scores) # N x C
        
        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1)    # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = np.transpose(np.transpose(expscores) / (sumexp))     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        # suma elemenata matrice logprobs tako da u retku i zbrojim član indeksiran s Y_[i]
        loss = -np.sum(logprobs[np.arange(X.shape[0]), Y_]) / X.shape[0]  # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Y_vec = np.zeros((X.shape[0], c))
        Y_vec[np.arange(X.shape[0]), Y_] = 1
        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_vec    # N x C

        # gradijenti parametara
        grad_W = np.dot(np.transpose(dL_ds), X) / X.shape[0]    # C x D
        grad_b = np.sum(dL_ds, axis=0) / X.shape[0]    # 1 x C

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def logreg_classify(X, W, b):
    scores = np.dot(X, np.transpose(W)) + b
    return np.argmax(scores, axis=1)

def logreg_decfun(w,b):
    def classify(X):
        return logreg_classify(X, w,b)
    return classify

#def get_probs(x):
    


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    w,b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    #probs = get_probs(np.dot(X, np.transpose(w)) + b)
    Y = logreg_classify(X, w,b)
    #Y = probs

    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    #AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision)

    # graph the decision surface
    decfun = logreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
  
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()