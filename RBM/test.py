import pandas as pd
import torch as T

from rbm import RBMNetwork

def main():
    df = pd.read_csv("test.csv")
    v_size = df.shape[1]
    h_size = 100
    rbm = RBMNetwork(v_size, h_size)
    lr = 0.01
    for i in range(len(df)):
        data = list(df.loc[i])
        data = T.tensor(data, dtype=T.float).to(rbm.device)
        if i == 0:
            rbm.init_cd(data)
        else:
            rbm.update(data, lr)
    mean = rbm.mean_x([1, 1, 0, 0, 0, 0, 0], [True, True, True, True, False, False, False])
    print(mean)
    pass

if __name__ == "__main__":
    main()