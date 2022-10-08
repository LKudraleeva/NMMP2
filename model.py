from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from mpmath import besselj, besseljzero, exp

R = 3.0
T = 80.0
L = 0.5
K = 0.13
C = 2.64
ALPHA = 0.002
n = 100


def psi(ri) -> float:
    if 0.0 <= ri <= 3.0 / 5.0:
        return 180.0
    else:
        return 0.0


def implicit_solution(r: [float],
                      t: [float]) -> [[float]]:
    _I, _K = len(r) - 1, len(t) - 1
    hr = r[1] - r[0]
    ht = t[1] - t[0]
    gamma = K * ht / (C * hr ** 2)
    betta = 2.0 * ALPHA * ht / (L * C)
    alpha_0 = 1.0 + 4.0 * gamma + betta
    alpha = 1.0 + 2.0 * gamma + betta

    u = np.zeros((_K + 1, _I + 1))

    for i in range(0, _I):
        u[0][i] = psi(r[i])

    p_0 = 4.0 * gamma / alpha_0
    q_0 = 1.0 / alpha_0

    for k in range(1, _K + 1):
        p, q = np.zeros(_I - 1), np.zeros(_I - 1)
        p[0] = p_0
        q[0] = q_0 * u[k - 1][0]

        for i in range(1, _I - 1):
            e = 1.0 - hr / r[i]
            d = alpha - gamma * hr / r[i] - gamma * p[i - 1] * e
            p[i] = gamma / d
            q[i] = (u[k - 1][i] + gamma * q[i - 1] * e) / d

        e = 1.0 - hr / r[_I - 1]
        d = alpha - gamma * hr / r[_I - 1] - gamma * p[_I - 2] * e
        u[k][_I - 1] = (u[k - 1][_I - 1] + gamma * q[_I - 2] * e) / d

        for i in range(_I - 2, -1, -1):
            u[k][i] = p[i] * u[k][i + 1] + q[i]

    return u


def analytic_solution(r, t, label):
    # Bessel zeros
    bessel_zeros = []
    for i in range(1, n + 1):
        bessel_zeros.append(besseljzero(0, i))
    u = []
    if label == 'r':
        for _t in t:
            v = 0
            for z in bessel_zeros:
                v += exp(-1 * _t * (K / C * pow(z / R, 2) + 2 * ALPHA / (L * C))) * 72 * besselj(1,
                                                                                                 z / 5) * besselj(
                    0, z * r / R) / (pow(besselj(1, z), 2) * z)
            u.append(v)
    else:
        for _r in r:
            v = 0
            for z in bessel_zeros:
                v += exp(-1 * t * (K / C * pow(z / R, 2) + 2 * ALPHA / (L * C))) * 72 * besselj(1, z / 5) * besselj(
                    0, z * _r / R) / (pow(besselj(1, z), 2) * z)
            u.append(v)
    return u


def plot(result, arr, arr2, values, label, cur_i, cur_k):
    h = arr[1] - arr[0]
    for value in values:
        k = ceil(value / h)
        if label == 'r':
            plt.plot(arr2, result[:, k], label='Разностное решение: ' + ' I = ' + str(cur_i) + ' K = ' + str(cur_k))
            plt.xlabel('t')
        else:
            plt.plot(arr2, result[k, :], label='Разностное решение: ' + ' I = ' + str(cur_i) + ' K = ' + str(cur_k))
            plt.xlabel('r')


if __name__ == '__main__':
    arr_I = [10, 20, 40, 80, 160, 320]
    arr_K = [20, 40, 80, 160, 320, 640]

    for _i, _k in zip(arr_I, arr_K):
        t_ar = np.linspace(0, T, num=_i)
        r_ar = np.linspace(0, R, num=_k)
        res = implicit_solution(r_ar, t_ar)
        t_values = [10.0]
        # r_values = [0.0, 1.0, 2.0, 3.0]
        plot(res, t_ar, r_ar, t_values, 't', _i, _k)
        if _i == 320:
            analytic_res = analytic_solution(r_ar, 10.0, 't')
            plt.plot(r_ar, analytic_res, label='Аналитическое решение: ' + ' I = ' + str(_i) + ' K = ' + str(_k))

    # t_values = [0.0, 1.0, 5.0, 10.0, 70.0]
    plt.xlabel('r')
    plt.ylabel("U(r, t)")
    plt.grid()
    plt.legend()
    plt.show()
    # plot(v, r_ar, t_ar, r_values, 'r')
