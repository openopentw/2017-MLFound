import numpy as np

def read_dat(fn):
    dat = np.loadtxt(fn)
    X_dat = dat[:,:-1]
    y_dat = dat[:,-1]
    return X_dat, y_dat

def append_X_with_1(X_dat):
    new_X_dat = np.ones((X_dat.shape[0], X_dat.shape[1]+1))
    new_X_dat[:,:-1] = X_dat
    return new_X_dat

def calc_mistakes(X_dat, y_dat, w):
    return np.sum(np.dot(X_dat, w) * y_dat <= 0)

def train(X_train, y_train, w):
    best_mistakes = calc_mistakes(X_train, y_train, w)
    iters = 0
    while iters < 50:
        i = np.random.randint(X_train.shape[0])
        if np.dot(X_train[i], w) * y_train[i] <= 0:
            iters += 1
            w += y_train[i] * X_train[i]
            mistakes = calc_mistakes(X_train, y_train, w)
            if mistakes < best_mistakes:
                best_w = np.array(w)
                best_mistakes = mistakes
    return w

def vali(X_test, y_test, best_w):
    mistakes = calc_mistakes(X_test, y_test, best_w)
    return mistakes / X_test.shape[0]

X_train, y_train = read_dat('./hw1_18_train.dat')
X_train = append_X_with_1(X_train)
X_test, y_test = read_dat('./hw1_18_test.dat')
X_test = append_X_with_1(X_test)

sum_error_rate = 0
for i in range(2000):
    np.random.seed(i)
    ori_w = np.zeros(X_train.shape[1], dtype=float)
    best_w = train(X_train, y_train, ori_w)
    error_rate = vali(X_test, y_test, best_w)
    sum_error_rate += error_rate
print(sum_error_rate / 2000)
