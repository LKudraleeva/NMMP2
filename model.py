from math import ceil

import numpy as np
import matplotlib.pyplot as plt


def psi(ri) -> float:
    if 0.0 <= ri <= 3.0 / 5.0:
        return 180.0
    else:
        return 0.0


def implicit_solution(r: [float],
                      t: [float],
                      c: float,
                      k: float,
                      a: float,
                      l: float) -> [[float]]:
    I, K = len(r) - 1, len(t) - 1
    hr = r[1] - r[0]
    ht = t[1] - t[0]
    gamma = k * ht / (c * hr ** 2)
    betta = 2.0 * a * ht / (l * c)
    alpha_0 = 1.0 + 4.0 * gamma + betta
    alpha = 1.0 + 2.0 * gamma + betta

    u = np.zeros((K + 1, I + 1))

    for i in range(0, I):
        u[0][i] = psi(r[i])

    p_0 = 4.0 * gamma / alpha_0
    q_0 = 1.0 / alpha_0

    for k in range(1, K + 1):
        p, q = np.zeros(I - 1), np.zeros(I - 1)
        p[0] = p_0
        q[0] = q_0 * u[k - 1][0]

        for i in range(1, I - 1):
            e = 1.0 - hr / r[i]
            d = alpha - gamma * hr / r[i] - gamma * p[i - 1] * e
            p[i] = gamma / d
            q[i] = (u[k - 1][i] + gamma * q[i - 1] * e) / d

        e = 1.0 - hr / r[I - 1]
        d = alpha - gamma * hr / r[I - 1] - gamma * p[I - 2] * e
        u[k][I - 1] = (u[k - 1][I - 1] + gamma * q[I - 2] * e) / d

        for i in range(I - 2, -1, -1):
            u[k][i] = p[i] * u[k][i + 1] + q[i]

    return u


def plot(result, arr, values, label):
    h = arr[1] - arr[0]
    for value in values:
        k = ceil(value / h)
        if label == 'r':
            plt.plot(arr, result[:, k], label=label + ' = ' + str(value))
        else:
            plt.plot(arr, result[k, :], label=label + ' = ' + str(value))

    plt.xlabel(label)
    plt.ylabel("U(r, t)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    R = 3.0
    T = 80.0
    n = 80
    t_ar = np.linspace(0, T, num=n)
    r_ar = np.linspace(0, R, num=n)

    v = implicit_solution(r_ar, t_ar, 2.64, 0.13, 0.002, 0.5)

    t_values = [0.0, 1.0, 5.0, 10.0, 70.0]
    r_values = [0.0, 1.0, 2.0, 3.0]
    plot(v, t_ar, t_values, 't')
    plot(v, r_ar, r_values, 'r')
