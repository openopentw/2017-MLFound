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

def test(w, x):
    return np.sign(w.dot(x.T))

def err(y, y_pred):
    return np.sum(y_pred != y) / y.size

def transform(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 3))
    new_x[:,:3] = x
    new_x[:,3] = x[:,1] * x[:,2]
    new_x[:,4] = x[:,1] ** 2
    new_x[:,5] = x[:,2] ** 2
    return new_x

N = 1000
seed = 1
x_dat, y = gen_data(N, seed)
x = np.ones((x_dat.shape[0], x_dat.shape[1] + 1))
x[:,1:] = x_dat

x = transform(x)

w_my = linear_regression(x, y)
y_my = test(w_my, x)

other_w = [np.array([-1, -0.05, 0.08, 0.13,  1.5,  1.5]),
           np.array([-1, -0.05, 0.08, 0.13,   15,  1.5]),
           np.array([-1,  -1.5, 0.08, 0.13, 0.05,  1.5]),
           np.array([-1,  -1.5, 0.08, 0.13, 0.05, 0.05]),
           np.array([-1, -0.05, 0.08, 0.13,  1.5,   15])]

errs = []
for w in other_w:
    other_y = test(w, x)
    errs += [err(y_my, other_y)]

print('errs =', errs)
print('argmin =', errs.index(min(errs)))
