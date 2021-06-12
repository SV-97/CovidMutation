from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import gamma, loggamma, comb
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, log
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
    base_poly = Poly(*((element_bound + 1)*[1]))
    print(len(base_poly.coeffs))
    return (base_poly ** size)


TOTAL_BOUND = 200
ELEMENT_BOUND = 1
SIZE = 100

poly = get_poly(ELEMENT_BOUND, SIZE)
print(poly)


def phi(element_bound, size, total_bound):
    print(f"{total_bound=}")
    return poly.coeffs[total_bound]


@non_negative
@clamp_below(0)
def rho(value, element_bound, size, total_bound):
    return phi(element_bound, size-1, total_bound - value)


poly_deg = ELEMENT_BOUND * SIZE + 1
print(f"{poly_deg=}, {poly.degree=}")


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
l1, = plt.plot(list(range(len(poly.coeffs))), poly.coeffs, "--")
#l2 = plt.scatter(list(range(len(poly.coeffs))), poly.coeffs)
ax.margins(x=0)
plt.gca().axhline(0)

axcolor = 'lightgoldenrodyellow'
ax_total_bound = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_element_bound = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_size = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)


slider_total_bound = Slider(ax_total_bound, "Total Bound",
                            0, 200, valinit=TOTAL_BOUND, valstep=1)
slider_element_bound = Slider(ax_element_bound, "Element Bound",
                              0, 200, valinit=ELEMENT_BOUND, valstep=1)
slider_size = Slider(ax_size, "Size", 0, 200, valinit=SIZE, valstep=1)


def update(val):
    print("updating")
    total_bound = int(slider_total_bound.val)
    element_bound = int(slider_element_bound.val)
    size = int(slider_size.val)

    poly = get_poly(element_bound, size)
    print(poly)

    def phi(element_bound, size, total_bound):
        print(f"{total_bound=}")
        return poly.coeffs[total_bound]

    @non_negative
    @clamp_below(0)
    def rho(value, element_bound, size, total_bound):
        return phi(element_bound, size-1, total_bound - value)

    c = [log(y) for y in poly.coeffs]

    poly_deg = element_bound * SIZE + 1
    x = list(range(len(poly.coeffs)))

    ax.axis([x[0], len(x), 0, max(c) + 1])
    l1.set_data(x, c)
    #l2.set_data(x, poly.coeffs)
    fig.canvas.draw_idle()


slider_total_bound.on_changed(update)
slider_element_bound.on_changed(update)
slider_size.on_changed(update)

plt.show()
