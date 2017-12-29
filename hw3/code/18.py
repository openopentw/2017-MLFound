import numpy as np

#########
# utils #
#########

def preprocess(x):
    def add_const(x):
        new_x = np.ones((x.shape[0], x.shape[1] + 1))
        new_x[:,1:] = x
        return new_x
    return add_const(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def grad_ein(w, x, y):
    return np.mean(sigmoid(-y*w.dot(x.T)).reshape(x.shape[0], 1) * (-y.reshape(x.shape[0], 1)*x), axis=0)

def update_w(w, x, y, eta):
    return w - eta * grad_ein(w, x, y)

################
# train & test #
################

def train(x, y, T=2000, eta=0.0001):
    x = preprocess(x)
    w = np.zeros(x.shape[1])
    for t in range(T):
        w = update_w(w, x, y, eta)
    return w

def test(w, x):
    x = preprocess(x)
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

def main():
    x_tra, y_tra = read_dat('./dat/hw3_train.dat')
    w = train(x_tra, y_tra)
    # w = train(x_tra, y_tra, eta=0.01)

    x_tes, y_tes = read_dat('./dat/hw3_test.dat')
    y_pred = test(w, x_tes)
    print('eout =', err(y_tes, y_pred))

if __name__ == "__main__":
    main()
