import numpy as np
from math import factorial
from itertools import count
import matplotlib.pyplot as plt

TOTAL_BOUND = 200
ELEMENT_BOUND = 4
SIZE = 100


@np.vectorize
def f(x):
    """
    ((1-x**(ELEMENT_BOUND + 1)) / (1-x))**SIZE
    """
    y = 1/(1-x)
    return y


def cauchy_differentiation(n, f, r=1, a=0, base=0):
    t = np.complex128(np.linspace(0, 2*np.pi, num=10000))
    gamma = base + r * (np.cos(t) + 1j * np.sin(t))
    dgamma = gamma[1] - gamma[0]
    # print(abs(dgamma))
    x = gamma.real
    y = gamma.imag
    plt.plot(x, y)
    plt.show()

    fak = factorial(n)

    @np.vectorize
    def integrand(z):
        a1 = f(z)
        a2 = (1 / r)**(n+1) * z**(n+1)
        #a2 = (z-a)**(n+1)
        #print(a1, a2)
        return fak * a1 / a2 * abs(dgamma)
    parts = integrand(gamma)
    # print(parts)
    # print(fak)
    return 1 * np.sum(parts) / (2j * np.pi)


for n in count(0):
    d_n = cauchy_differentiation(n, np.exp, 1, 0, 0)
    print(n, d_n, int(d_n.real), int(abs(d_n)))


def myDerivative(f, x, dTheta, degree):
    riemannSum = 0
    theta = 0
    while theta < 2*np.pi:
        functionArgument = np.complex128(x + np.exp(1j*theta))
        secondFactor = np.complex128(np.exp(-1j * degree * theta))
        riemannSum += f(functionArgument) * secondFactor * dTheta
        theta += dTheta
    return factorial(degree)/(2*np.pi) * riemannSum.real
