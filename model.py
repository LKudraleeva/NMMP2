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
n = 50


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

    for i in range(0, _I + 1):
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


def plot(result, arr, arr2, values, label, cur_i=None, cur_k=None):
    h = arr[1] - arr[0]
    for value in values:
        k = ceil(value / h)
        if label == 'r':
            if cur_i:
                plt.plot(arr2, result[:, k], label='Разностное решение: ' + ' I = ' + str(cur_i) + ' K = ' + str(cur_k))
            else:
                plt.plot(arr2, result[:, k], label='r = ' + str(value))
            plt.xlabel('t')
        else:
            if cur_i:
                plt.plot(arr2, result[k, :], label='Разностное решение: ' + ' I = ' + str(cur_i) + ' K = ' + str(cur_k))
            else:
                plt.plot(arr2, result[k, :], label='t = ' + str(value))
            plt.xlabel('r')


def convergence():
    arr_i = [10, 20, 40, 80, 160, 320]
    arr_k = [20, 40, 80, 160, 320, 640]
    for _i, _k in zip(arr_i, arr_k):
        arr_t = np.linspace(0, T, num=_i)
        arr_r = np.linspace(0, R, num=_k)
        res = implicit_solution(arr_r, arr_t)
        r = 1.0
        plot(res, arr_r, arr_t, [r], 'r', _i, _k)
        if _i == 320:
            analytic_res = analytic_solution(1.0, arr_t, 'r')
            plt.plot(arr_t, analytic_res, label='Аналитическое решение: ' + ' I = ' + str(_i) + ' K = ' + str(_k))
    plt.title("r = 1.0")
    plt.xlabel('t')
    plt.ylabel("U(r, t)")
    plt.grid()
    plt.legend()
    plt.show()

    for _i, _k in zip(arr_i, arr_k):
        arr_t = np.linspace(0, T, num=_i)
        arr_r = np.linspace(0, R, num=_k)
        res = implicit_solution(arr_r, arr_t)
        t = 10.0
        plot(res, arr_t, arr_r, [t], 't', _i, _k)
        if _i == 320:
            analytic_res = analytic_solution(arr_r, 10.0, 't')
            plt.plot(arr_r, analytic_res, label='Аналитическое решение: ' + ' I = ' + str(_i) + ' K = ' + str(_k))
    plt.title("t = 10.0")
    plt.xlabel('r')
    plt.ylabel("U(r, t)")
    plt.grid()
    plt.legend()
    plt.show()


def visualization():
    t_values = [0.0, 1.0, 5.0, 10.0]
    r_values = [0.0, 1.0, 2.0, 3.0]
    t_ar = np.linspace(0, T, num=160)
    r_ar = np.linspace(0, R, num=320)
    res = implicit_solution(r_ar, t_ar)
    plot(res, r_ar, t_ar, r_values, 'r')
    plt.ylabel("U(r, t)")
    plt.grid()
    plt.legend()
    plt.show()
    plot(res, t_ar, r_ar, t_values, 't')
    plt.ylabel("U(r, t)")
    plt.grid()
    plt.legend()
    plt.show()


def accuracy():
    arr_i = [10, 20, 40, 80, 160, 320, 640, 1280]
    arr_k = [20, 40, 80, 160, 320, 640, 1280, 2560]
    impl_results = []
    analytic_res = 0
    for _i, _k in zip(arr_i, arr_k):
        arr_t = np.linspace(0, T, num=_i)
        arr_r = np.linspace(0, R, num=_k)
        t = 40.0
        k = ceil(t / (arr_t[1] - arr_t[0]))
        impl_results.append(implicit_solution(arr_r, arr_t)[k, 0])
        if _i == 320:
            analytic_res = analytic_solution(arr_r, 40.0, 't')[0]

    a = impl_results[0] - analytic_res
    for i in range(1, len(arr_i)):
        b = impl_results[i] - analytic_res
        c = a / b
        print(arr_i[i], arr_k[i], a, b, c)
        a = b


if __name__ == '__main__':
    # visualization()
    # convergence()
    accuracy()
