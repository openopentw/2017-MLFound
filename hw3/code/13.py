import numpy as np

def gen_data(N, seed):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (N, 2))
    y = np.sign(x[:,0]**2 + x[:,1]**2 - 0.6)
    y *= np.sign(np.random.uniform(0, 1, y.size) - 0.1)
    return x, y

def linear_regression(x, y):
    return np.linalg.pinv(x).dot(y)
    # return np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

N = 1000
eins = []
eouts = []
for seed in range(1000):
    x_dat, y = gen_data(N, seed)
    x = np.ones((x_dat.shape[0], x_dat.shape[1] + 1))
    x[:,1:] = x_dat

    w = linear_regression(x, y)
    y_pred = np.sign(w.dot(x.T))
    ein = np.sum(y_pred != y) / y.size
    eins += [ein]

    test_x_dat, test_y = gen_data(N, seed*7122)
    test_x = np.ones((test_x_dat.shape[0], test_x_dat.shape[1] + 1))
    test_x[:,1:] = test_x_dat
    y_pred = np.sign(w.dot(test_x.T))
    eout = np.sum(y_pred != test_y) / test_y.size
    eouts += [eout]

print('E_in =', sum(eins) / len(eins))
print('E_out =', sum(eouts) / len(eouts))
