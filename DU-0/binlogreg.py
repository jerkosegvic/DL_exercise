import numpy as np
import matplotlib.pyplot as plt
import random
import data

 # stabilni softmax
"""
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs
"""

def binlogreg_decfun(w,b):
    def classify(X):
        return binlogreg_classify(X, w,b)
    return classify

def get_probs(x):
    exp_x = np.exp(x)
    probs = exp_x / (1 + exp_x)
    return probs

def calc_loss(X, Y_, probs):
    # gubitak
    loss = -np.sum(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs)) / X.shape[0]
    return loss

def binlogreg_train(X,Y_):
    '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w, b: parametri logističke regresije
    '''

    # TODO 1: inicijalizirati parametre modela
    w = np.random.randn(X.shape[1])
    b = np.zeros(1)

    # TODO 2: postaviti parametre algoritma
    param_niter = 400
    param_delta = 0.01

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, np.transpose(w) ) + b    # N x 1
        
        # vjerojatnosti razreda c_1
        probs = get_probs(scores)   # N x 1

        # gubitak
        loss = calc_loss(X, Y_, probs)  # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_     # N x 1
        
        # gradijenti parametara
        grad_w = np.dot(dL_dscores, (X)) / X.shape[0]     # D x 1
        grad_b = np.sum( dL_dscores ) / X.shape[0]     # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    '''
    Argumenti
      X:  podatci, np.array NxD
      w, b: parametri logističke regresije

    Povratne vrijednosti
      probs: vjerojatnosti razreda c_1, np.array Nx1
    '''

    # TODO 3: izračunati vjerojatnosti razreda c_1
    scores = np.dot(X, np.transpose(w) ) + b    # N x 1
    probs = get_probs(scores)   # N x 1

    return probs 


if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w,b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = probs > 0.5

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
  
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()