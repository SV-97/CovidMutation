
from itertools import count, product
import random
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Optional
import re


def markov_1(start: List[Nucleobase], end: List[Nucleobase], alpha: float, beta: float, t_max=1_000_000, n_samples=10_000):
    end = np.array(end)
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

    base = (A, C, G, T)

    def sample_distribution(dist):
        return random.choices(base, weights=dist, k=1)[0]

    # take n_samples samples from the end distribution
    samples = np.array([np.apply_along_axis(
        sample_distribution, axis=1, arr=total_end_distribution) for _ in range(n_samples)])

    # check if sample state equals end state
    def is_end(sample):
        return np.all(sample == end)

    # calculate proportion of samples that are in the end state
    return np.sum(np.apply_along_axis(is_end, axis=1, arr=samples)) / n_samples


def markov_2(start: List[Nucleobase], end: List[Nucleobase], alpha: float, beta: float, t_max=1_000_000, n_samples=10_000):
    # get a sample of possible mutated sequences
    samples = markov_sample(start, alpha, beta, t_max, n_samples)

    # calculate proportion of samples that are in the end state
    return np.sum(np.apply_along_axis(is_equivalent_to_end, axis=1, arr=samples)) / n_samples


# NEWER OLD


class Nucleobase(Enum):
    __slots__ = "A", "C", "G", "T"
    A = 0
    C = 1
    G = 2
    T = 3

    def __str__(self):
        return {A: "A", C: "C", G: "G", T: "T"}[self]
    __repr__ = __str__


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
CODON_TABLE = {m[0]: m[1] for m in matches}


def equivalent(s1, s2):
    return CODON_TABLE[s1] == CODON_TABLE[s2]


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
        print(seq_to_str(state))
        indices = random.choices(
            index_range, k=random.randint(0, max_mutations_per_step))
        for index_to_change in indices:
            if random.uniform(0, 1) < 1/9:
                state[index_to_change] = transversion(state[index_to_change])
            else:
                state[index_to_change] = transition(state[index_to_change])
            if state == end:
                return i


def sample_markov(start: List[Nucleobase], alpha: float, beta: float, t_max, n_samples):
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

    base = (A, C, G, T)

    def sample_distribution(dist):
        return random.choices(base, weights=dist, k=1)[0]

    # take n_samples samples from the end distribution
    samples = np.array([np.apply_along_axis(
        sample_distribution, axis=1, arr=total_end_distribution) for _ in range(n_samples)])

    return samples


def simulate_markov(start: List[Nucleobase], end_predicate: Callable[[np.array], bool], alpha: float, beta: float, t_max=1_000_000, n_samples=10_000):
    samples = sample_markov(start, alpha, beta, t_max, n_samples)

    # calculate proportion of samples that are in the end state
    return np.sum(np.apply_along_axis(end_predicate, axis=1, arr=samples)) / n_samples


# generate all equivalent sequences
methionin = "ATG",
prolin = "CCT", "CCC", "CCA", "CCG"
arginin = "CGT", "CGC", "CGA", "CGG", "AGA", "AGG"
alanin = "GCT", "GCC", "GCA", "GCG"
stop_codon = "TAG", "TGA", "TAA"
seq = (methionin, prolin, arginin, arginin, alanin, stop_codon)


def possible_codes(seq):
    return set(map(tuple, product(*seq)))


equivalent_sequences = {np.array(str_to_seq("".join(s))).tostring()
                        for s in set(possible_codes(seq))}


def is_equivalent_to_end(sample):
    """ Check if sample state is equivalent to some end state """
    return sample.tostring() in equivalent_sequences


end = str_to_seq("ATG-CCT-CGG-CGG-GCA-TAG")


def is_equal_to_end(sample):
    return np.all(sample == end)


if __name__ == "__main__":
    random.seed(0)
    start = str_to_seq("TAG-TAG-TAG-TAG-TAG-TAG")
    # max_mutations_per_step = len(start)
    # print(simulate(start, end, max_mutations_per_step)

    alpha = 0.04 / 100  # transition
    beta = 0.005 / 100  # transversion
    print(simulate_markov(start, is_equivalent_to_end,
                          alpha, beta, t_max=1_000_000, n_samples=1_000_000))
