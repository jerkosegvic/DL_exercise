import torch.nn as nn
import torch
import numpy as np
import data
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
        - D: dimensions of each datapoint 
        - C: number of classes
        """

        super(PTLogreg, self).__init__()
             
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        self.w = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.randn(C))

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        s = torch.mm(X, self.w) + self.b
        return torch.softmax(s, dim=1)

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...        
        return torch.mean(torch.sum(-Yoh_ * torch.log(self.forward(X)), dim=1))

def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    # ...
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_)

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()




def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()    
    return model.forward(torch.Tensor(X)).detach().numpy()

def predict(model, X):
    return np.argmax(eval(model, X), axis=1)

def logreg_decfun(model):
    def classify(X):
        return predict(model, X)
    return classify

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gauss_2d(2, 100)
    Yoh_ = data.class_to_onehot(Y_)
    Xt = torch.Tensor(X)
    Yoht_ = torch.Tensor(Yoh_)

    # definiraj model:
    ptlr = PTLogreg(Xt.shape[1], Yoht_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, Xt, Yoht_, 1000, 0.5)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, Xt)
    Y = predict(ptlr, Xt)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    #AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision)
    #breakpoint()
    # iscrtaj rezultate, decizijsku plohu
    decfun = logreg_decfun(ptlr)
    data.graph_surface(decfun, (np.min(X, axis=0), np.max(X, axis=0)), offset=0.5)
    data.graph_data(Xt, Y_, Y, special=[])

    # show the plot
    plt.show()