from scipy.special import gamma, loggamma, comb
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from functools import wraps
from poly import Poly

np.seterr("raise")


def non_negative(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        x = f(*args, **kwargs)
        if x < 0:
            raise ValueError(f"Encountered negative value {x} for args "
                             f"{args}, {kwargs}, function= {f}")
        else:
            return x
    return wrap


def clamp_below(lower_bound):
    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            x = f(*args, **kwargs)
            if x < lower_bound:
                return lower_bound
            else:
                return x
        return wrap
    return decorator


def psi(size, total_bound):
    """
    if size == 0:
        return 1
    if total_bound < 0:
        return 0
    assert(size >= 1)
    assert(total_bound >= 0)
    return factorial(total_bound+size-1) / \
        (factorial(total_bound) * factorial(size-1))
    """
    return comb(size + total_bound - 1, total_bound, exact=True)


def phi(element_bound, size, total_bound):
    # print(element_bound, size, total_bound)
    if size == 0:
        return 1 if total_bound == 0 else 0
    if total_bound <= element_bound:
        return psi(size, total_bound)
    assert(size >= 1)
    assert(total_bound >= 0)
    assert(total_bound >= element_bound + 1)
    return psi(size, total_bound) - size * \
        psi(size, total_bound - (element_bound + 1))


@non_negative
@clamp_below(0)
def rho(value, element_bound, size, total_bound):
    #print(value, element_bound, size, total_bound)
    assert(0 <= value <= total_bound and size >= 1)
    if total_bound / size > element_bound:
        return 0  # can't form such a partition
    elif total_bound / size == element_bound:
        return 1 if value == element_bound else 0
    if element_bound == 0:
        if total_bound == 0 and value == 0:
            return 1
        else:
            return 0
    if size == 1:
        return 1 if size == value else 0
    else:  # size > 1
        if total_bound == value:
            return 1
        else:
            return phi(element_bound, size-1, total_bound - value)


def get_poly(element_bound, size):
    base_poly = Poly(*(element_bound*[1]))
    return (base_poly ** size)


TOTAL_BOUND = 20000
ELEMENT_BOUND = 2
SIZE = 50

poly = get_poly(ELEMENT_BOUND, SIZE)
print(poly)


def phi(element_bound, size, total_bound):
    return poly.coeffs[total_bound + 1]


@non_negative
@clamp_below(0)
def rho(value, element_bound, size, total_bound):
    return phi(element_bound, size-1, total_bound - value)


fig = plt.figure()
plt.gca().axhline(0)
# reversed(range(TOTAL_BOUND // SIZE, ELEMENT_BOUND + 1)):
for element_bound in [ELEMENT_BOUND]:
    #element_bound = 3
    print(f"n={TOTAL_BOUND}, r={element_bound}, k={SIZE}")
    vals = list(range(len(poly.coeffs)))
    counts = poly.coeffs
    #counts = [rho(m, element_bound, SIZE, TOTAL_BOUND) for m in vals]
    print(sum(counts))
    counts = np.array(counts) / sum(counts)
    ax = plt.gca()
    ax.plot(
        vals, counts)
    ax.scatter(
        vals, counts, label=f"n={TOTAL_BOUND}, r={element_bound}, k={SIZE}", )
    ax.set_yscale("log")
plt.legend()
plt.show()
