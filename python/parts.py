from collections import Counter
from functools import lru_cache, reduce
from math import prod  # , factorial
import matplotlib.pyplot as plt
import random as rng
import numpy as np
from itertools import product
from numpy.lib.ufunclike import isneginf
from scipy.special import gamma, loggamma

np.seterr("raise")


def factorial(x):
    return gamma(x+1)


def logfactorial(x):
    return loggamma(x+1)


@lru_cache
def phi(r, k, n):
    """
    x = (factorial(n+k-1) / (factorial(n)) - k *
         factorial(n-r+k-2)/factorial(n-r-1))/factorial(k-1)
    return x"""
    a = logfactorial(n+k-1) - logfactorial(n) - logfactorial(k-1)
    b = np.log(k) + logfactorial(n-r+k-2) - \
        logfactorial(n-r-1) - logfactorial(k-1)
    print(a, b)
    def exp(x): return 1 + x**2/2 + x**3/6 + x**4/24 + x**5 / 120
    return a - (0 if np.isneginf(b) or np.isnan(b) else exp(b))


"""    if a > 700 or b > 700:
        # return np.exp(np.abs(a-b)/2) * (a-b) * (1 + max(a, b))
        return exp(a) - (0 if np.isneginf(b) or np.isnan(b) else exp(b))
    else:
        return np.exp(a) - (0 if np.isneginf(b) or np.isnan(b) else np.exp(b))"""


def psi(k, n):
    return factorial(n+k-1) / (factorial(n) * factorial(k-1))


def P_x_is_m(m, r, k, n):
    if m > r:
        return 0
    # / (psi(k, n) - (k-1) * sum(psi(k-1, m) for m in range(0, n-r)))
    try:
        return phi(r, k-1, n-m) / phi(r, k, n)
    except ValueError as e:
        print(e)
        print(
            f"{m=}, {r=}, {k=}, {n=}, {k-1=}, {n-m=}, {n-m+k-1-1=}, {n-m-r+k-1-2=}, {n-m-r-1=}")
        return 0


def sigma(r, n, k):
    return psi(k, n) - (k-1) * sum(psi(k-1, m) for m in range(0, n-r))


n = 10
r = 4
k = 7
# Number of bounded partitions:  52374  number of all partitions:  53130
# phi(r,k,n) und psi(k,n) stimmen
#parts = [x for x in product(*(range(0, r+1) for _ in range(k))) if sum(x) == n]
# all_parts = [x for x in product(*(range(0, n+1)
#                                  for _ in range(k))) if sum(x) == n]
#print("Number of bounded partitions: ", len(parts))
#print(" number of all partitions: ", len(all_parts))
x = np.array(range(0, r+1))
m = np.array([P_x_is_m(m, r, k, n) for m in x])
x = x[~np.isnan(m)]
m = m[~np.isnan(m)]
s = sum(m)
m /= s
print(m)
print("Summe: ", s, sum(m))
# print(s, sigma(r, n, k))
#print(phi(r, k, n), psi(k, n), psi(k, n) - psi(k, n-r))
plt.plot(x, m)
plt.scatter(x, m)
plt.show()

"""def parts(n, k):
    return prod((n+i)/i for i in range(1, k))


parts(6, 3)


def p(r, n, m):
    return prod(1-r/(n+i) for i in range(1, m-1)) / (n + (m - 1))


def sample(target_density, M):
    def phi(x):
        return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    u = rng.random()
    y = rng.normalvariate(0, 1)
    while u >= target_density(y) / (M * phi(y)):
        y = rng.normalvariate(0, 1)
    return y


def bla(m, n, k):
    partition = []
    x = n
    for i in range(k):
        r = rng.randrange(x//m, k)
        x -= r
        partition.append(r)
    return partition


n = 38728123
m = 71231
x = list(range(n+1))
y = [(m-1) * p(r, n, m) for r in x]


def bla(m, n, _):
    while True:
        r = rng.choices(x, weights=y, k=m)
        if sum(r) == n:
            return r


parts = [tuple(sorted(bla(6, 16, 3))) for _ in range(100)]


def fff(c, part):
    c.update(part)
    return c


brr = reduce(fff, parts, Counter())
xs, ys = zip(*brr.items())
plt.scatter(list(xs), list(ys))
print(brr)
plt.show()

# print([int(parts(n, m)) for m in range(1, n+1)])
# for m in range(0, n+1):
x = list(range(n+1))
y = [(m-1) * p(r, n, m) for r in x]
# print(m, sum(y), y)
plt.plot(x, y, label=f"{m}")
plt.scatter(x, y)
plt.legend()
plt.show()
"""
