import numpy as np

def gen_data(N, seed):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (N, 2))
    y = np.sign(x[:,0]**2 + x[:,1]**2 - 0.6)
    y *= np.sign(np.random.uniform(0, 1, y.size) - 0.1)
    return x, y

def preprocess(x):
    def add_const(x):
        new_x = np.ones((x.shape[0], x.shape[1] + 1))
        new_x[:,1:] = x
        return new_x

    def transform(x):
        new_x = np.zeros((x.shape[0], x.shape[1] + 3))
        new_x[:,:3] = x
        new_x[:,3] = x[:,1] * x[:,2]
        new_x[:,4] = x[:,1] ** 2
        new_x[:,5] = x[:,2] ** 2
        return new_x

    return transform(add_const(x))

def linear_regression(x, y):
    return np.linalg.pinv(x).dot(y)
    # return np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

def test(w, x):
    return np.sign(w.dot(x.T))

def err(y, y_pred):
    return np.sum(y_pred != y) / y.size

N = 1000
seed = 1

x, y = gen_data(N, seed)

x = preprocess(x)
my_w = linear_regression(x, y)

errs = np.zeros(N)
for seed in range(N):
    test_x, test_y = gen_data(N, seed*7122)
    test_x = preprocess(test_x)

    my_y = test(my_w, test_x)
    my_err = err(test_y, my_y)
    errs[seed] = my_err

print(errs.mean(0))
