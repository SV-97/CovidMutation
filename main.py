from time import time
from numba import njit
import re
from typing import Any, Callable, Collection, Dict, Hashable, List, Mapping, Optional, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
import random
from itertools import count, product
from functools import partial


class Nucleobase(Enum):
    __slots__ = "A", "C", "G", "T"
    A = 0
    C = 1
    G = 2
    T = 3

    def __str__(self):
        return {A: "A", C: "C", G: "G", T: "T"}[self]
    __repr__ = __str__


@np.vectorize
def to_value(nuc: Nucleobase):
    return nuc.value


A = Nucleobase.A
C = Nucleobase.C
G = Nucleobase.G
T = Nucleobase.T


def str_to_seq(s):
    return [getattr(Nucleobase, n) for n in s if n in ("A", "C", "G", "T")]


def seq_to_str(s):
    return "".join(str(n) for n in s)


def transition(s):
    if s == A:
        return G
    if s == G:
        return A
    if s == C:
        return T
    if s == T:
        return C


def transversion(s):
    if s in (A, G):
        return random.choice((C, T))
    if s in (C, T):
        return random.choice((A, G))


with open("codon.txt", "r") as f:
    raw_codon = f.read()
matches = re.findall(
    r"(?P<abbrev>[UCGA]{3})\s+(?P<full_name>(\w|ä)+)", raw_codon)
CODON_TABLE = {m[0].replace("U", "T"): m[1] for m in matches}


def equivalent(s1, s2):
    return CODON_TABLE[s1] == CODON_TABLE[s2]


def preimage(x, d: Mapping) -> Set:
    """Find preimage of x in d"""
    p = {key for key, value in d.items() if value == x}
    if len(p) == 0:
        raise ValueError(f"No values for {x} found in dictionary {d}")
    return p


def simulate(start, end, max_mutations_per_step):
    """
    8 p_transversion = p_transition
    and p_transition + p_transversion = 1
    <=> p_transversion = 1/9
    """
    state = start[:]
    len_state = len(state)
    index_range = range(len_state)
    for i in count(1):
        indices = random.choices(
            index_range, k=random.randint(0, max_mutations_per_step))
        for index_to_change in indices:
            if random.uniform(0, 1) < 1/9:
                state[index_to_change] = transversion(state[index_to_change])
            else:
                state[index_to_change] = transition(state[index_to_change])
            if state == end:
                return i


def sample_markov(start: List[Nucleobase], alpha: float, beta: float, t_max, n_samples, rng=np.random.default_rng(0)):
    gamma = 1 - 2*beta - alpha
    p_single = np.array([
        [gamma, beta, alpha, beta],
        [beta, gamma, beta, alpha],
        [alpha, beta, gamma, beta],
        [beta, alpha, beta, gamma],
    ])
    p_single_end = np.linalg.matrix_power(p_single, t_max)

    I = np.identity(4)
    start_distribution = np.array([I[base.value, :] for base in start])
    total_end_distribution = start_distribution @ p_single_end

    base = np.array((A.value, C.value, G.value, T.value))
    samples = np.array([rng.choice(base, n_samples, p=distribution)
                        for distribution in total_end_distribution]).transpose()
    return samples


def simulate_markov(start: List[Nucleobase], end_predicate: Callable[[np.array], bool], alpha: float, beta: float, t_max=1_000_000, n_samples=10_000):
    samples = sample_markov(start, alpha, beta, t_max, n_samples)
    # samples = np.vectorize(lambda n: n.value)(samples)
    # calculate proportion of samples that are in the end state
    return np.sum(np.apply_along_axis(end_predicate, axis=1, arr=samples)) / n_samples


def find_equivalence_class(sequence: Collection[str], d: Mapping) -> Set[Tuple]:
    seq = list(preimage(s, d) for s in sequence)
    return set(tuple(str_to_seq("".join(s))) for s in product(*seq))


# generate all equivalent sequences
equiv = find_equivalence_class(
    ("Methionin", "Prolin", "Arginin", "Arginin", "Alanin", "Stop"), CODON_TABLE)
EQUIV = np.array([np.array(e) for e in equiv])
EQUIV = to_value(EQUIV)
EQUIV_HASH = {hash(tuple(row)) for row in EQUIV}


def is_equivalent_to_end(sample, comp_set=EQUIV_HASH):
    """ Check if sample state is equivalent to some end state """
    return hash(tuple(sample)) in comp_set


end = np.array(str_to_seq("ATG-CCT-CGG-CGG-GCA-TAG"))
end_val = to_value(end)


@njit
def is_equal_to_end(sample):
    return np.all(sample == end_val)


if __name__ == "__main__":
    random.seed(0)
    start = np.array(str_to_seq("TAG-TAG-TAG-TAG-TAG-TAG"))
    # max_mutations_per_step = len(start)
    # print(simulate(start, end, max_mutations_per_step)

    alpha = 0.04 / 100  # transition
    beta = 0.005 / 100  # transversion
    t1 = time()
    res = simulate_markov(start, is_equivalent_to_end,
                          alpha, beta, t_max=1_000_000, n_samples=10_000_000)
    t2 = time()
    print(f"Mutation {start} -> {end}\nhas p={res}")
    print(f"\nRan in {t2 - t1}s")
