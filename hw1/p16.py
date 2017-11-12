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

def random_shuffle(X_dat, y_dat, seed):
    shuffled_list = np.arange(X_dat.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_list)
    shuffled_X_dat = X_dat[shuffled_list]
    shuffled_y_dat = y_dat[shuffled_list]
    return shuffled_X_dat, shuffled_y_dat

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

sum_update_cnt = 0
for i in range(2000):
    w = np.zeros(X_dat.shape[1], dtype=float)
    shuffled_X_dat, shuffled_y_dat = random_shuffle(X_dat, y_dat, i)
    update_cnt = train(shuffled_X_dat, shuffled_y_dat, w)
    # print(update_cnt)
    sum_update_cnt += update_cnt
print(sum_update_cnt / 2000)
