#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![allow(dead_code, unused_variables)]
mod mat;

use mat::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
// use rand::rngs::SmallRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use regex::Regex;
use static_init::dynamic;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::io::prelude::*;
use std::str::FromStr;
use std::thread;

const N_WORKER_THREADS: usize = 1;

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
enum Nucleobase {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
}

impl fmt::Display for Nucleobase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Nucleobase::A => 'A',
                Nucleobase::C => 'C',
                Nucleobase::G => 'G',
                Nucleobase::T => 'T',
            }
        )
    }
}

impl Default for Nucleobase {
    fn default() -> Self {
        Nucleobase::A
    }
}

impl FromStr for Nucleobase {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.chars().next().unwrap() {
            'A' => Ok(Nucleobase::A),
            'C' => Ok(Nucleobase::C),
            'G' => Ok(Nucleobase::G),
            'U' | 'T' => Ok(Nucleobase::T),
            _ => Err("Invalid nucleotide"),
        }
    }
}

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
struct Codon((Nucleobase, Nucleobase, Nucleobase));

impl Codon {
    pub fn a(&self) -> Nucleobase {
        self.0 .0
    }
    pub fn b(&self) -> Nucleobase {
        self.0 .1
    }
    pub fn c(&self) -> Nucleobase {
        self.0 .2
    }
}

impl Default for Codon {
    fn default() -> Self {
        Codon((
            Nucleobase::default(),
            Nucleobase::default(),
            Nucleobase::default(),
        ))
    }
}

impl FromStr for Codon {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut ch = s.chars();
        let a = Nucleobase::from_str(&ch.next().unwrap().to_string()).unwrap();
        let b = Nucleobase::from_str(&ch.next().unwrap().to_string()).unwrap();
        let c = Nucleobase::from_str(&ch.next().unwrap().to_string()).unwrap();

        if ch.next().is_none() {
            Ok(Codon((a, b, c)))
        } else {
            Err("Too many elements in codon string.")
        }
    }
}

impl fmt::Display for Codon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}{}", self.0 .0, self.0 .1, self.0 .2)
    }
}

fn read_codon_table() -> HashMap<Codon, String> {
    let mut file = std::fs::File::open("../codon.txt").expect("Failed to open file.");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Failed to read file.");
    let re = Regex::new(r"(?P<abbrev>[UCGA]{3})\s+(?P<full_name>\w+)").unwrap();
    let mut h = HashMap::new();
    for m in re.captures_iter(&content) {
        let cod = Codon::from_str(&m["abbrev"]).unwrap();
        let full_name = &m["full_name"];
        h.insert(cod, full_name.to_owned());
    }
    h
}

fn preimage<X: Copy + Hash + Eq, Y: Eq>(y: &Y, d: &HashMap<X, Y>) -> HashSet<X> {
    let mut s = HashSet::new();
    for (key, val) in d {
        if val == y {
            s.insert(*key);
        }
    }
    s
}

type Sequence = [Codon; 6];

fn parse_sequence(s: &str) -> Sequence {
    let mut v = Vec::with_capacity(6);
    for triple in s.split('-') {
        v.push(Codon::from_str(triple).unwrap());
    }
    v.try_into().unwrap()
}

fn cartesian_product(v: &[Vec<Codon>]) -> Vec<Vec<Codon>> {
    if v.len() == 1 {
        let mut r = vec![];
        for x in &v[0] {
            r.push(vec![*x]);
        }
        r
    } else {
        let mut r = vec![];
        for x in &v[0] {
            for mut y in cartesian_product(&v[1..]) {
                y.insert(0, *x);
                r.push(y);
            }
        }
        r
    }
}

fn print_seq(codons: &[Codon]) {
    let s = codons.iter().fold(None, |acc, c| match acc {
        None => Some(c.to_string()),
        Some(mut s) => {
            s.push('-');
            s.push_str(&c.to_string());
            Some(s)
        }
    });
    println!("{}", s.unwrap());
}

fn total_end_distribution(start: Sequence, alpha: f64, beta: f64, t_max: usize) -> Mat<18, 4> {
    let start_distribution = {
        let mut m = Mat::<18, 4>::zero();
        for (i, codon) in start.iter().enumerate() {
            let j = 3 * i;
            m[(j, codon.a() as usize)] = 1.0;
            m[(j + 1, codon.b() as usize)] = 1.0;
            m[(j + 2, codon.c() as usize)] = 1.0;
        }
        m
    };
    start_distribution * probability_matrix(alpha, beta, t_max)
}

fn probability_matrix(alpha: f64, beta: f64, t_max: usize) -> Mat<4, 4> {
    let gamma = 1. - 2. * beta - alpha;
    let p_single = Mat::new([
        [gamma, beta, alpha, beta],
        [beta, gamma, beta, alpha],
        [alpha, beta, gamma, beta],
        [beta, alpha, beta, gamma],
    ]);
    p_single.powi(t_max)
}

fn count_in_distribution_sample(
    end_predicate: impl Fn(&Sequence) -> bool,
    distributions: Mat<18, 4>,
    n_samples: usize,
    mut rng: impl Rng,
) -> usize {
    let base = [Nucleobase::A, Nucleobase::C, Nucleobase::G, Nucleobase::T];
    let mut count = 0;
    for _ in 0..n_samples {
        let mut seq = [Codon::default(); 6];
        let mut f = |nu| {
            //*base.choose(&mut rng).unwrap()
            *base
                .choose_weighted(&mut rng, |item| distributions[(nu, *item as usize)])
                .unwrap()
        };
        for (j, s) in seq.iter_mut().enumerate() {
            let nu = j * 3;
            *s = Codon((f(nu), f(nu + 1), f(nu + 2)));
        }
        if end_predicate(&seq) {
            count += 1;
        }
    }
    count
}

/// Approximates ln(Γ(x)) using an asymptotic expansion based on Stirling's formula
fn log_gamma(x: f64) -> f64 {
    (x - 0.5) * x.ln() - x
        + 0.5 * (2. * std::f64::consts::PI).ln()
        + 1. / (12. * x)
        + 1. / (360. * x.powi(3))
}

/// Approximates ln(x!)
fn log_fac_f(x: f64) -> f64 {
    log_gamma(x + 1.)
}

fn log_fac(n: usize) -> f64 {
    let mut x = 0.;
    for i in 1..=n {
        x += (i as f64).ln();
    }
    x
}

fn fac(n: usize) -> usize {
    let mut x = 1;
    for i in 1..=n {
        x *= i;
    }
    x
}

/// If φ(r, k, n) denotes the number of k-partitions of n such that all elements in the partition are <= r
/// this function computes an approximation of φ(r, k-1, n-m) / φ(r, k, n); so this computes the probability
/// that one element of the partition has the value of m.
/// Note that because of the approximations the a distribution generated using this function may not be normalized
/// We have to renormalize the distribution because we work with approximations.
/// e.g.:
/// let s = distribution.iter().sum::<f64>();
/// for x in distribution.iter_mut() {
///     *x /= s;
/// }
fn bounded_k_partition_ratio(m: usize, r: usize, k: usize, n: usize) -> f64 {
    fn normalize(x: f64) -> f64 {
        if x.is_nan() || x.is_infinite() || x < 0. {
            0.
        } else {
            x
        }
    }
    let phi = |r: usize, n: usize, k: usize| -> f64 {
        let a = normalize(log_gamma((n + k) as f64 - 1.) - log_fac(n) - log_gamma((k as f64) - 1.));
        let b = if n < r
            || n < m
            || n < m + r
            || n + k < r + 1
            || n + k < 1
            || n + k < m + r + 2
            || n + k < m + 2
            || k == 1
        {
            0.
        } else {
            normalize(
                (k as f64).ln() + log_fac_f((n + k) as f64 - (2 + r) as f64)
                    - log_fac_f(n as f64 - (r + 1) as f64)
                    - log_fac(k - 1),
            )
        };
        if a < 700. && b < 700. {
            a.exp() - b.exp()
        } else {
            a - b
        }
    };

    if k < 1 || n < m {
        0.
    } else {
        phi(r, k - 1, n - m) / phi(r, k, n)
    }
}

/// If φ(r, k, n) denotes the number of k-partitions of n such that all elements in the partition are <= r
/// this function computes an approximation of φ(r, k-1, n-m) - which acts as a stand-in for the probability
/// that one element of the partition has the value of m that is given by φ(r, k-1, n-m) / φ(r, k, n).
fn bounded_k_partitions(m: usize, r: usize, k: usize, n: usize) -> usize {
    fn normalize(x: f64) -> f64 {
        if x.is_nan() || x.is_infinite() || x < 0. {
            0.
        } else {
            x
        }
    }
    let phi = |r: usize, n: usize, k: usize| -> usize {
        let a = normalize(log_gamma((n + k) as f64 - 1.) - log_fac(n) - log_gamma((k as f64) - 1.));
        let b = if n < r
            || n < m
            || n < m + r
            || n + k < r + 1
            || n + k < 1
            || n + k < m + r + 2
            || n + k < m + 2
            || k == 1
        {
            0.
        } else {
            normalize(
                (k as f64).ln() + log_fac_f((n + k) as f64 - (2 + r) as f64)
                    - log_fac_f(n as f64 - (r + 1) as f64)
                    - log_fac(k - 1),
            )
        };
        if a < 700. && b < 700. {
            (a.exp() - b.exp()) as usize
        } else {
            (a as usize).saturating_sub(b as usize)
        }
    };
    if k < 1 || n < m {
        0
    } else {
        phi(r, k - 1, n - m)
    }
}

/// Generate random m-partition of n where each entry is <= k
/// To generate a normal m-partition of n with this set k=n.
fn random_partition<Rng: RngCore>(n: usize, m: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    if k * m < n {
        panic!(
            "Can't form a {}-partition of {} where each element is <= {}",
            m, n, k
        );
    }
    let mut partition = Vec::with_capacity(m);
    let mut remainder = n;
    loop {
        partition.clear();
        for i in 1..m {
            // We first generate the probability distribution of our random partition
            let mut weights = Vec::with_capacity(n + 1);
            for i in 0..=std::cmp::min(k, remainder) {
                weights.push(bounded_k_partition_ratio(i, k, m, remainder));
            }
            let distribution = WeightedIndex::new(&weights).unwrap();
            loop {
                let r = distribution.sample(rng);
                if r <= remainder {
                    remainder = remainder.saturating_sub(r);
                    partition.push(r);
                    break;
                }
            }
        }
        partition.push(remainder);
        if *partition.iter().max().unwrap() <= k {
            break;
        } else {
            println!("shit, {}", *partition.iter().max().unwrap());
        }
    }
    // partition.shuffle(rng);
    partition
}

/// Find a possible end distribution
fn possible_distribution<Rng: RngCore>(
    start: Sequence,
    alpha: f64,
    beta: f64,
    t_max: usize,
    n_simulataneous_mutations: usize,
    rng: &mut Rng,
) -> Mat<18, 4> {
    if n_simulataneous_mutations > 18 {
        panic!("There can be at most 18 simultaneous mutations because there are only 18 codons.")
    }
    let start_distribution = {
        let mut m = Mat::<18, 4>::zero();
        for (i, codon) in start.iter().enumerate() {
            let j = 3 * i;
            m[(j, codon.a() as usize)] = 1.0;
            m[(j + 1, codon.b() as usize)] = 1.0;
            m[(j + 2, codon.c() as usize)] = 1.0;
        }
        m
    };
    //println!("{}", start_distribution);
    /*
    Consider the case where we have n_simultaneous_mutations = 1 - that is:
    exactly one codon in our sequence mutates in each step. Denote the number
    of mutations in codon k by c[k]. Then the sum over k of c[k] is equal to
    t_max. That means: c=(c[0], c[1], ..., c[7]) is an 8-partition of t_max,
    where each c[k] is trivially bounded above by t_max.
    Now consider the general case where we have n_simultaneous_mutations ∈ ℕ
    and at each step we have exactly n_simultaneous_mutations mutations,
    so in total we have t_max ⋅ n_simultaneous_mutations mutations - but it still
    holds that each codon can mutate at most once per each step so we still have
    c[k] <= t_max. Combined this means that c is an 8-partition of
    t_max ⋅ n_simultaneous_mutations where c[k] <= t_max.
    If now only have *up to* n_simultaneous_mutations mutations per step, then
    we have x <= t_max ⋅ n_simultaneous_mutations mutations for some random x > 0
    and c is an 8-partition of x where c[k] <= t_max.
    */
    let x = rng.gen_range(0..=t_max * n_simulataneous_mutations);
    let partition = random_partition(x, 18, t_max, rng);
    println!("Assuming {} mutations distributed as {:?}", x, partition);
    let mut total_end_dist = Mat::zero();
    for (i, c) in partition.into_iter().enumerate() {
        let p = probability_matrix(alpha, beta, c);
        total_end_dist.set_row(i, start_distribution.row(i) * p);
    }
    // println!("Distribution for t={}: {}", t_max, total_end_dist);
    total_end_dist
}

fn simulate_markov<
    Rng: SeedableRng + RngCore,
    F: Fn(&Sequence) -> bool + 'static + Send + Clone,
>(
    start: Sequence,
    end_predicate: F,
    alpha: f64,
    beta: f64,
    t_max: usize,
    n_samples: usize,
    n_simulataneous_mutations: usize,
) -> f64 {
    let mut master_rng = Rng::seed_from_u64(N_WORKER_THREADS as u64);

    let mut threads = vec![];
    for i in 0..N_WORKER_THREADS {
        // find one possible probability distribution
        let distribution = possible_distribution(
            start,
            alpha,
            beta,
            t_max,
            n_simulataneous_mutations,
            &mut master_rng,
        );
        let n_samples_thread = n_samples / N_WORKER_THREADS;
        let thread_pred = end_predicate.clone();
        threads.push(thread::spawn(move || {
            count_in_distribution_sample(
                thread_pred,
                distribution,
                n_samples_thread,
                &mut Rng::seed_from_u64(i as u64),
            )
        }));
    }
    let mut count: usize = threads.into_iter().map(|t| t.join().unwrap()).sum();
    let distribution = possible_distribution(
        start,
        alpha,
        beta,
        t_max,
        n_simulataneous_mutations,
        &mut master_rng,
    );
    // process all samples that aren't processed on the worker threads
    count += count_in_distribution_sample(
        end_predicate,
        distribution,
        n_samples - N_WORKER_THREADS * (n_samples / N_WORKER_THREADS),
        &mut master_rng,
    );
    (count as f64) / (n_samples as f64)
}

use Nucleobase::*;
const START: Sequence = [Codon((T, A, G)); 6];
#[dynamic]
static END: Sequence = parse_sequence("ATG-CCT-CGG-CGG-GCA-TAG");
#[dynamic]
static CODON_TABLE: HashMap<Codon, String> = read_codon_table();

#[dynamic]
static EQUIVALENCE_CLASSES: HashMap<String, HashSet<Codon>> = {
    let mut h = HashMap::new();
    for val in CODON_TABLE.values() {
        if !h.contains_key(val) {
            h.insert(val.clone(), preimage(val, &CODON_TABLE));
        }
    }
    h
};

#[dynamic]
static END_EQUIV: Vec<Sequence> = {
    // equivalence class of end sequence
    let mut s = vec![];
    for codon in END.iter() {
        let name = &CODON_TABLE[&codon];
        s.push(
            EQUIVALENCE_CLASSES[name]
                .iter()
                .copied()
                .collect::<Vec<Codon>>(),
        );
    }
    let cp = cartesian_product(&s[..]);
    //dbg!(cp.len());
    cp.into_iter().map(|x| x.try_into().unwrap()).collect()
};

fn main() {
    let t1 = std::time::Instant::now();

    let h = [0; 100_000_000]
        .iter()
        .map(|_| random_partition(20, 4, 10, &mut rand::thread_rng()))
        .fold(HashMap::<[usize; 4], usize>::new(), |mut h, p| {
            let mut p: [usize; 4] = p.try_into().unwrap();
            //p.sort();
            let count = h.entry(p).or_insert(0);
            *count += 1;
            h
        });
    for (key, value) in h {
        println!("{:?} : {}", key, value);
    }
    let phi = |r: usize, n: usize, k: usize| -> usize {
        let a = log_gamma((n + k) as f64 - 1.) - log_fac(n) - log_gamma((k as f64) - 1.);
        let b = (k as f64).ln() + log_fac_f((n + k) as f64 - (2 + r) as f64)
            - log_fac_f(n as f64 - (r + 1) as f64)
            - log_fac(k - 1);
        if a < 700. && b < 700. {
            (a.exp() - b.exp()) as usize
        } else {
            (a as usize).saturating_sub(b as usize)
        }
    };
    dbg!(phi(4, 10, 4));
    panic!();

    let alpha = 0.04 / 100.0; // transition
    let beta = 0.005 / 100.0; // transversion

    let is_equal_to_end = |x: &Sequence| x == &START;
    let is_equivalent_to_end = |x: &Sequence| END_EQUIV.contains(x);

    // Mit Gleichheit und 800_000_000 samples: p = 0
    // Mit Äquivalenz und 800_000_000 samples: p = 0.00000004125
    // Could also use e.g. SmallRng

    random_partition(10, 4, 7, &mut rand::thread_rng());

    let p = simulate_markov::<Xoshiro256PlusPlus, _>(
        START,
        is_equivalent_to_end,
        alpha,
        beta,
        1_000_000,
        100_000_000,
        1,
    );
    println!("p = {}", p);
    let t2 = std::time::Instant::now();
    println!("Total Runtime: {:#?}", t2 - t1);
}
