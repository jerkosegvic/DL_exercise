import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

#X = torch.tensor([1, 2])
#Y = torch.tensor([3, 5])
X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = torch.tensor([3, 5, 7, 9, 11.01, 12.99, 15, 16.99, 19, 21])


# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(1000):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.sum(diff**2) / X.shape[0]

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')