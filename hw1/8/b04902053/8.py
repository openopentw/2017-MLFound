import matplotlib.pyplot as plt
import numpy as np
import sys

def read_dat(fn):
    dat = np.loadtxt(fn)
    X_dat = dat[:,:-1]
    y_dat = dat[:,-1]
    return X_dat, y_dat

def append_X_with_1(X_dat):
    new_X_dat = np.ones((X_dat.shape[0], X_dat.shape[1]+1))
    new_X_dat[:,:-1] = X_dat
    return new_X_dat

def random_shuffle(X_dat, y_dat):
    shuffled_list = np.arange(X_dat.shape[0])
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
                update_cnt += 1
                is_full_cycle = False
    return update_cnt

def __main__():
    # read argvs
    if len(sys.argv) != 3:
        print('Usage: python3 8.py <data set> <output png>')
        print('e.g.: python3 8.py hw1_8_train.dat output.png')
        return
    print('Load data from:', sys.argv[1])    # It should be './hw1_8_train.dat'.
    print('Will output png file to:', sys.argv[2])
    fn_dat = sys.argv[1]    # It should be './hw1_8_train.dat'.
    fn_png = sys.argv[2]

    # read data & preprocess
    X_dat, y_dat = read_dat(fn_dat)
    X_dat = append_X_with_1(X_dat)

    # train 2000 times
    update_cnts = []
    for i in range(2000):
        np.random.seed(i)
        w = np.zeros(X_dat.shape[1], dtype=float)
        shuffled_X_dat, shuffled_y_dat = random_shuffle(X_dat, y_dat)
        update_cnt = train(shuffled_X_dat, shuffled_y_dat, w)
        update_cnts += [update_cnt]
    update_cnts = np.array(update_cnts)

    # print average number of updates
    print('average number of updates:', np.mean(update_cnts))

    # plot histogram
    plt.hist(update_cnts, bins=np.arange(0, update_cnts.max()+2, 1))
    plt.title('Number of Updates vs. Frequency')
    plt.xlabel('Number of Updates')
    plt.ylabel('Frequency')
    plt.savefig(fn_png)

if __name__ == '__main__':
    __main__()
