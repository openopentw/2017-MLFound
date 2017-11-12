import numpy as np

def read_dat(fn):
    dat = np.loadtxt(fn)
    X_dat = dat[:,:-1]
    y_dat = dat[:,-1]
    return X_dat, y_dat

def append_X0_with_1(X_dat):
    new_X_dat = np.ones((X_dat.shape[0], X_dat.shape[1]+1))
    new_X_dat[:,:-1] = X_dat
    return new_X_dat

def train(X_dat, y_dat, w):
    update_cnt = 0

    is_full_cycle = False
    while not is_full_cycle:
        is_full_cycle = True
        for i in range(X_dat.shape[0]):
            if np.dot(X_dat[i], w) * y_dat[i] <= 0:
                w += y_dat[i] * X_dat[i]
                # print(i, w)
                update_cnt += 1
                is_full_cycle = False

    return update_cnt

X_dat, y_dat = read_dat('./hw1_8_train.dat')
X_dat = append_X0_with_1(X_dat)
w = np.zeros(X_dat.shape[1], dtype=float)

update_cnt = train(X_dat, y_dat, w)
print(update_cnt)
