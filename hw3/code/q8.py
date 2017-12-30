import matplotlib.pyplot as plt
import numpy as np

#########
# utils #
#########

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def grad_ein(w, x, y, stochastic):
    if stochastic:
        return sigmoid(-y*w.dot(x.T)) * (-y*x)
    else:
        return np.mean(sigmoid(-y*w.dot(x.T)).reshape(x.shape[0], 1) * (-y.reshape(x.shape[0], 1)*x), axis=0)

def update_w(w, x, y, eta, stochastic):
    return w - eta * grad_ein(w, x, y, stochastic)

################
# train & test #
################

def train(w, x, y, eta, stochastic):
    return w - eta * grad_ein(w, x, y, stochastic)

def test(w, x):
    return np.sign(w.dot(x.T))

def err(y, y_pred):
    return np.mean(y != y_pred)

########
# main #
########

def read_dat(dat_fn):
    dat = np.genfromtxt(dat_fn)
    x_dat = dat[:,:-1]
    y_dat = dat[:,-1]
    return x_dat, y_dat

def add_const(x):
    new_x = np.ones((x.shape[0], x.shape[1] + 1))
    new_x[:,1:] = x
    return new_x

def main():
    x_tra, y_tra = read_dat('./dat/hw3_train.dat')
    x_tra = add_const(x_tra)

    T = 2000

    # lr = 0.001

    def gd_sgd_plt(lr, plt_fn):
        # GD
        eins_001_gd = []
        w = np.zeros(x_tra.shape[1])
        for t in range(T):
            w = train(w, x_tra, y_tra, lr, False)
            y_pred = test(w, x_tra)
            eins_001_gd += [err(y_tra, y_pred)]

        # SGD
        eins_001_sgd = []
        w = np.zeros(x_tra.shape[1])
        for t in range(T):
            choose = t % x_tra.shape[0]
            w = train(w, x_tra[choose], y_tra[choose], lr, True)
            y_pred = test(w, x_tra)
            eins_001_sgd += [err(y_tra, y_pred)]

        # plot histogram
        plt.rcParams['font.family'] = 'serif'

        plt.plot(range(T), eins_001_gd,  'C1', label='GD')
        plt.plot(range(T), eins_001_sgd, 'C2', label='SGD')
        plt.xlabel('Iteratinos')
        plt.ylabel('Ein')
        plt.legend()
        plt.savefig(plt_fn)
        plt.close()

    gd_sgd_plt(0.001, 'q8_001')
    gd_sgd_plt(0.01, 'q8_01')

if __name__ == "__main__":
    main()
