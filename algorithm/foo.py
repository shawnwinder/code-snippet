import numpy as np
import sys, math

def get_grads(X):
    N = len(X)
    M = len(X[0])
    coef_grads = np.ones(M)

    for i in range(M - 1):
        i += 1
        coef_grads[i] = np.sum(X[:,i]) / N

    return coef_grads

def calc_mse(X, Y, coefs):
    Y_hat = np.sum(X * coefs)
    mse = np.square(Y_hat - Y)
    return mse

def solve(X, Y):
    N = len(X)
    M = len(X[0])

    coefs = np.random.randn(M)
    X = np.array(X)
    Y = np.array(Y)

    MSE = 0
    for i in range(N):
        MSE += calc_mse(X[i], Y[i], coefs)

    eps = 1e-3
    eta = 1e-1
    last_mse = MSE
    iters = 0
    while True or iters < 100000:
        # gradient descent bp
        coef_grads = get_grads(X)
        coefs = coefs - eta*coef_grads

        # calculate mse
        new_mse = 0
        for i in range(N):
            new_mse += calc_mse(X[i], Y[i], coefs)

        # stop bp
        if np.abs(new_mse - last_mse) < eps:
            break

        last_mse = new_mse
        ++ iters
        print("===")
        print("last mse: {}".format(last_mse))
        print("new mse: {}".format(new_mse))

    return coefs.tolist()

if __name__ == "__main__":
    X = []
    Y = []

    while True:
        line = sys.stdin.readline().strip('\r\n')
        if line == '':
            break

        nums = line.split(',')
        nums = [float(n) for n in nums]
        X.append([1.0] + nums[:-1])
        Y.append(nums[-1])

    coefs = solve(X, Y)

    formatted_coefs = []
    for coef in coefs:
        coef = math.floor(coef * 100) / 100
        formatted_coefs.append('%.2f' % coef)

    print(','.join(formatted_coefs))
