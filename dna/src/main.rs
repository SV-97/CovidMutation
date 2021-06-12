#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![allow(dead_code, unused_variables)]
mod mat;
mod partition;
mod polynomial;

use mat::*;
use partition::random_partition;
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

const N_WORKER_THREADS: usize = 8;
/// Number of nucleobases in a sequence
const N_NUCLEOBASES: usize = 6;
const SEQUENCE_LENGTH: usize = N_NUCLEOBASES / 3;
/// A new mutation-distribution is created after this many samples
const GENERATE_NEW_DIST_EVERY_N: usize = 1_000_000;
const SHOW_MESSAGES: bool = false;

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

type Sequence = [Codon; SEQUENCE_LENGTH];

fn parse_sequence(s: &str) -> Sequence {
    let mut v = Vec::with_capacity(SEQUENCE_LENGTH);
    for triple in s.split('-') {
        v.push(Codon::from_str(triple).unwrap());
    }
    v.try_into().unwrap()
}

fn cartesian_product(v: &[Vec<Codon>]) -> Vec<Vec<Codon>> {
    let mut r = vec![];
    if v.len() == 1 {
        for x in &v[0] {
            r.push(vec![*x]);
        }
    } else {
        for x in &v[0] {
            for mut y in cartesian_product(&v[1..]) {
                y.insert(0, *x);
                r.push(y);
            }
        }
    }
    r
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

fn total_end_distribution(
    start: Sequence,
    alpha: f64,
    beta: f64,
    t_max: usize,
) -> Mat<N_NUCLEOBASES, 4> {
    let start_distribution = {
        let mut m = Mat::<N_NUCLEOBASES, 4>::zero();
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

fn count_in_distribution_sample<Rng: RngCore>(
    end_predicate: impl Fn(&Sequence) -> bool,
    gen_distributions: impl Fn(&mut Rng) -> Mat<N_NUCLEOBASES, 4>,
    n_samples: usize,
    mut rng: &mut Rng,
) -> usize {
    let base = [Nucleobase::A, Nucleobase::C, Nucleobase::G, Nucleobase::T];
    let mut count = 0;
    let mut distributions = gen_distributions(rng);
    for i in 0..n_samples {
        if i % GENERATE_NEW_DIST_EVERY_N == 0 {
            distributions = gen_distributions(rng);
        }
        let mut seq = [Codon::default(); SEQUENCE_LENGTH];
        let mut f = |nu| {
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

#[derive(Clone, Copy, PartialEq, Debug)]
struct StartDistParams {
    pub alpha: f64,
    pub beta: f64,
    pub start: Sequence,
}

fn find_possible_distribution<Rng: RngCore>(
    start_params: &StartDistParams,
    t_max: usize,
    n_simulataneous_mutations: usize,
    rng: &mut Rng,
) -> Mat<N_NUCLEOBASES, 4> {
    let start = start_params.start;
    let alpha = start_params.beta;
    let beta = start_params.beta;
    if n_simulataneous_mutations > N_NUCLEOBASES {
        panic!(
            "There can be at most {} simultaneous mutations because there are only {} codons.",
            N_NUCLEOBASES, N_NUCLEOBASES
        )
    }
    let start_distribution = {
        let mut m = Mat::<N_NUCLEOBASES, 4>::zero();
        for (i, codon) in start.iter().enumerate() {
            let j = 3 * i;
            m[(j, codon.a() as usize)] = 1.0;
            m[(j + 1, codon.b() as usize)] = 1.0;
            m[(j + 2, codon.c() as usize)] = 1.0;
        }
        m
    };
    let (x, partition) =
        random_partition(N_NUCLEOBASES, t_max, t_max * n_simulataneous_mutations, rng);

    if SHOW_MESSAGES {
        println!("Assuming {} mutations distributed as {:?}", x, partition);
    }
    let mut total_end_dist = Mat::zero();
    for (i, c) in partition.into_iter().enumerate() {
        let p = probability_matrix(alpha, beta, c);
        total_end_dist.set_row(i, start_distribution.row(i) * p);
    }
    //println!("{}", total_end_dist);
    // println!("Distribution for t={}: {}", t_max, total_end_dist);
    total_end_dist
}

fn simulate_markov<
    Rng: SeedableRng + RngCore,
    F: Fn(&Sequence) -> bool + 'static + Send + Clone,
>(
    end_predicate: F,
    start_params: StartDistParams,
    t_max: usize,
    n_samples: usize,
    n_simulataneous_mutations: usize,
    seed: usize,
) -> f64 {
    let mut master_rng = Rng::seed_from_u64((seed + N_WORKER_THREADS) as u64);
    let mut threads = vec![];
    let gen_distribution = move |rng: &mut Rng| {
        find_possible_distribution(&start_params, t_max, n_simulataneous_mutations, rng)
    };
    for i in 0..N_WORKER_THREADS {
        // find one possible probability distribution
        let n_samples_thread = n_samples / N_WORKER_THREADS;
        let thread_pred = end_predicate.clone();
        threads.push(thread::spawn(move || {
            let mut rng = Rng::seed_from_u64((seed + i) as u64);
            count_in_distribution_sample(thread_pred, gen_distribution, n_samples_thread, &mut rng)
        }));
    }
    let mut count: usize = threads.into_iter().map(|t| t.join().unwrap()).sum();
    // process all samples that aren't processed on the worker threads
    count += count_in_distribution_sample(
        end_predicate,
        gen_distribution,
        n_samples - N_WORKER_THREADS * (n_samples / N_WORKER_THREADS),
        &mut master_rng,
    );
    count as f64 / n_samples as f64
}

use Nucleobase::*;
const START: Sequence = [Codon((T, A, G)); SEQUENCE_LENGTH];
#[dynamic]
//static END: Sequence = parse_sequence("ATG-CCT-CGG-CGG-GCA-TAG");
static END: Sequence = parse_sequence("ATG-CCT");
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
static END_EQUIV: HashSet<Sequence> = {
    // equivalence class of end sequence
    let mut s = vec![];
    for codon in END.iter() {
        let name = &CODON_TABLE[codon];
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

    let start_params = StartDistParams {
        start: START,
        alpha: 0.04 / 100.0, // transition
        beta: 0.005 / 100.0, // transversion
    };

    let is_equal_to_end = |x: &Sequence| x == &START;
    let is_equivalent_to_end = |x: &Sequence| END_EQUIV.contains(x);

    let seed = 0;
    // Could also use e.g. SmallRng rather than Xoshiro
    let p = simulate_markov::<Xoshiro256PlusPlus, _>(
        is_equivalent_to_end,
        start_params,
        100,
        800_000_000,
        N_NUCLEOBASES,
        seed,
    );
    println!("p = {}", p);
    let t2 = std::time::Instant::now();
    println!("Total Runtime: {:#?}", t2 - t1);
}

/*
Code for testing uniformity of partition generators
let mut h = [0; 100_000_000]
        .iter()
        .map(|_| random_partition_2(4, 5, 10, rng).1)
        .fold(HashMap::<[usize; 4], usize>::new(), |mut h, p| {
            let mut p: [usize; 4] = p.try_into().unwrap();
            let count = h.entry(p).or_insert(0);
            *count += 1;
            h
        });
    for (key, value) in h {
        println!("{:?} : {}", key, value);
    }
    panic!();
*/
