import matplotlib.pyplot as plt
import numpy as np

def gen_data(seed):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, 20)
    y = np.sign(x)
    y *= np.sign(np.random.uniform(0,1,20) - 0.2)
    return x, y

def train(x, y):
    argsort = np.argsort(x)
    x = np.array(x[argsort])
    y = np.array(y[argsort])
    # print(y)

    lowest_Ein = len(x)
    lowest_theta = []
    lowest_s = []

    num = len(x)

    for theta in range(num):
        Ein = np.sum(y[:theta] == 1) + np.sum(y[theta:] == -1)
        if Ein < lowest_Ein:
            lowest_Ein = Ein
            lowest_theta = [theta]
            lowest_s = [1]
        elif Ein == lowest_Ein:
            lowest_theta += [theta]
            lowest_s += [1]

    for theta in range(num):
        Ein = np.sum(y[:theta] == -1) + np.sum(y[theta:] == 1)
        if Ein < lowest_Ein:
            lowest_Ein = Ein
            lowest_theta = [theta]
            lowest_s = [-1]
        elif Ein == lowest_Ein:
            lowest_theta += [theta]
            lowest_s += [-1]

    choice = np.random.choice(len(lowest_s))
    lowest_s = lowest_s[choice]
    lowest_theta = lowest_theta[choice]

    return lowest_Ein / 20, lowest_s, (lowest_theta-10) / 20

Eins = []
Eouts = []
for i in range(1000):
    x, y = gen_data(i)
    lowest_Ein, lowest_s, lowest_theta = train(x, y)
    Eins += [lowest_Ein]
    Eout = 0.5 + 0.3 * lowest_s * (np.fabs(lowest_theta) - 1)
    Eouts += [Eout]
# print(sum(Eins) / 1000)
# print(sum(Eouts) / 1000)

plt.scatter(Eins, Eouts)
plt.title('Ein vs. Eout')
plt.xlabel("Ein")
plt.ylabel("Eout")
plt.savefig('17.png')
