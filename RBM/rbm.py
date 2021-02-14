import random

import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import utils
from torch.distributions.binomial import Binomial

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1 / np.sqrt(fan_in)
    return (-lim, lim)

class RBMNetwork(nn.Module):
    INIT_SIGMA = 1e-0

    def __init__(self, v_size, h_size):
        super().__init__()
        self.v_size = v_size
        self.h_size = h_size
        self.device = T.device("cuda:0")
        self.reset_parameters()
        self.to(self.device)
    
    def reset_parameters(self):
        self.bv = T.randn((self.v_size,), dtype=T.float).to(self.device) * self.INIT_SIGMA
        self.bh = T.randn((self.h_size,), dtype=T.float).to(self.device) * self.INIT_SIGMA
        self.W = T.randn((self.v_size, self.h_size), dtype=T.float).to(self.device) * self.INIT_SIGMA

    def prob_x(self, h):
        energy = T.mm(self.W, T.reshape(h, (-1, 1))).reshape((-1,))
        energy = energy + self.bv
        prob_x = F.sigmoid(energy)
        return prob_x
    
    def prob_h(self, x):
        energy = T.mm(T.reshape(x, (1, -1)), self.W).reshape((-1,))
        energy = energy + self.bh
        prob_h = F.sigmoid(energy)
        return prob_h
    
    def sample_x(self, h):
        prob_x = self.prob_x(h)
        bist = Binomial(probs=prob_x)
        x = bist.sample()
        return x
    
    def sample_h(self, x):
        prob_h = self.prob_h(x)
        bist = Binomial(probs=prob_h)
        h = bist.sample()
        return h

    def init_cd(self, x):
        self.x_cd = x

    def update(self, x_sample, lr):
        h_sample = self.sample_h(x_sample)
        h_cd = self.sample_h(self.x_cd)
        dbv = x_sample - self.x_cd
        dbh = h_sample - h_cd
        dW = T.matmul(x_sample.reshape(-1, 1), h_sample.reshape(1, -1))\
                - T.matmul(self.x_cd.reshape(-1, 1), h_cd.reshape(1, -1))
        self.bv = self.bv + lr * dbv
        self.bh = self.bh + lr * dbh
        self.W = self.W + lr * dW

        self.x_cd = self.sample_x(h_cd)
    
    def mean_x(self, x_sample, mask):
        x_sample = T.tensor(x_sample, dtype=T.float).to(self.device)
        mask = T.tensor(mask, dtype=T.float).to(self.device)
        x = x_sample
        xs = []
        for _ in range(1000):
            x = x_sample * mask + x * T.logical_not(mask)
            h = self.sample_h(x)
            x = self.sample_x(h)
            xs.append(x)
        mean = T.mean(T.stack(xs), axis=0)
        return mean

def test():
    v_size = 8
    h_size = 20
    rbm = RBMNetwork(v_size, h_size)
    lr = 0.01
    omega = [[0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1, 1]]
    n_train = 10000
    x_sample = random.choice(omega)
    x_sample = T.tensor(x_sample, dtype=T.float).to(rbm.device)
    rbm.init_cd(x_sample)
    for i in range(n_train):
        x_sample = random.choice(omega)
        x_sample = T.tensor(x_sample, dtype=T.float).to(rbm.device)
        rbm.update(x_sample, lr)
    x_sample = [1, 1, 0, 0, 0, 0, 0, 0]
    mask = [True, True, True, True, False, False, False, False]
    mean = rbm.mean_x(x_sample, mask)
    print(mean)
        
    


def main():
    test()

if __name__ == "__main__":
    main()