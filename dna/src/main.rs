#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![allow(dead_code, unused_variables)]
mod mat;
mod partitions;

use mat::*;
use partitions::random_partition;
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
const N_NUCLEOBASES: usize = 6;
const SEQUENCE_LENGTH: usize = N_NUCLEOBASES / 3;

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

fn count_in_distribution_sample(
    end_predicate: impl Fn(&Sequence) -> bool,
    distributions: Mat<N_NUCLEOBASES, 4>,
    n_samples: usize,
    mut rng: impl Rng,
) -> usize {
    let base = [Nucleobase::A, Nucleobase::C, Nucleobase::G, Nucleobase::T];
    let mut count = 0;
    for _ in 0..n_samples {
        let mut seq = [Codon::default(); SEQUENCE_LENGTH];
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

/// Find a possible end distribution
fn possible_distribution<Rng: RngCore>(
    start: Sequence,
    alpha: f64,
    beta: f64,
    t_max: usize,
    n_simulataneous_mutations: usize,
    rng: &mut Rng,
) -> Mat<N_NUCLEOBASES, 4> {
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
    //println!("{}", start_distribution);
    /*
    Consider the case where we have n_simultaneous_mutations = 1 - that is:
    exactly one codon out of the 6*3=18 codons in our sequence mutates in each step.
    Denote the number of mutations in codon k by c[k]. Then the sum over k of c[k]
    is equal to t_max. That means: c=(c[0], c[1], ..., c[17]) is an 18-partition of
    t_max, where each c[k] is trivially bounded above by t_max.
    Now consider the general case where we have n_simultaneous_mutations ∈ ℕ
    and at each step we have exactly n_simultaneous_mutations mutations,
    so in total we have t_max ⋅ n_simultaneous_mutations mutations - but it still
    holds that each codon can mutate at most once per each step so we still have
    c[k] <= t_max. Combined this means that c is an 18-partition of
    t_max ⋅ n_simultaneous_mutations where c[k] <= t_max.
    If now only have *up to* n_simultaneous_mutations mutations per step, then
    we have x <= t_max ⋅ n_simultaneous_mutations mutations for some random x > 0
    and c is an 18-partition of x where c[k] <= t_max.
    */
    let x = rng.gen_range(0..=t_max * n_simulataneous_mutations);
    let partition = random_partition(x, N_NUCLEOBASES, t_max, rng);
    println!("Assuming {} mutations distributed as {:?}", x, partition);
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
    start: Sequence,
    end_predicate: F,
    alpha: f64,
    beta: f64,
    t_max: usize,
    n_samples: usize,
    n_simulataneous_mutations: usize,
    seed: usize,
) -> f64 {
    let mut master_rng = Rng::seed_from_u64((seed + N_WORKER_THREADS) as u64);

    let mut threads = vec![];
    for i in 0..N_WORKER_THREADS {
        // find one possible probability distribution
        let n_samples_thread = n_samples / N_WORKER_THREADS;
        let thread_pred = end_predicate.clone();
        threads.push(thread::spawn(move || {
            let mut rng = Rng::seed_from_u64((seed + i) as u64);
            let distribution = possible_distribution(
                start,
                alpha,
                beta,
                t_max,
                n_simulataneous_mutations,
                &mut rng,
            );
            count_in_distribution_sample(thread_pred, distribution, n_samples_thread, &mut rng)
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
    dbg!(count as f64) / dbg!(n_samples as f64)
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
/* This should be a HashSet */
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

    let alpha = 0.04 / 100.0; // transition
    let beta = 0.005 / 100.0; // transversion

    let is_equal_to_end = |x: &Sequence| x == &START;
    let is_equivalent_to_end = |x: &Sequence| END_EQUIV.contains(x);

    // Mit Gleichheit und 800_000_000 samples: p = 0
    // Mit Äquivalenz und 800_000_000 samples: p = 0.00000004125
    /*
    Äquivalenz:
        1_000_000 Schritte:
           Mit 18 Mutationen pro Schritt und 800_000_000 samples:
           p = 0.00000002875
           Total Runtime: 262.214484813s

           Mit einer Mutation pro Schritt und 800_000_000 samples:
           p = 0.0000000275
           Total Runtime: 249.62549916s

    Gleichheit:
        1_000_000 Schritte:
            Mit 18 Mutationen pro Schritt und 800_000_000 samples:
            p = 0
            Total Runtime: 102.068615662s

            Mit einer Mutation pro Schritt und 800_000_000 samples:
            p = 0.0000000475
            Total Runtime: 102.203317899s
    */

    let seed = 0;
    // Could also use e.g. SmallRng rather than Xoshiro
    let p = simulate_markov::<Xoshiro256PlusPlus, _>(
        START,
        is_equivalent_to_end,
        alpha,
        beta,
        1_000_000,
        100_000_000,
        N_NUCLEOBASES,
        seed,
    );
    println!("p = {}", p);
    let t2 = std::time::Instant::now();
    println!("Total Runtime: {:#?}", t2 - t1);
}

/*
    Searching for 18-partition of 5842355 such that each element is <= 1000000
    Searching for 18-partition of 9668929 such that each element is <= 1000000
    Searching for 18-partition of 11662516 such that each element is <= 1000000
    Searching for 18-partition of 12634971 such that each element is <= 1000000
    Searching for 18-partition of 12232306 such that each element is <= 1000000
    Searching for 18-partition of 1802716 such that each element is <= 1000000
    Searching for 18-partition of 11005910 such that each element is <= 1000000
    Searching for 18-partition of 12916371 such that each element is <= 1000000
    Assuming 5842355 mutations distributed as [636295, 474987, 84813, 480247, 268830, 417094, 207841, 321117, 409801, 305592, 352697, 306631, 167247, 315776, 54250, 54727, 763227, 221183]
    Assuming 9668929 mutations distributed as [996390, 625579, 488030, 626437, 943880, 696279, 305845, 864693, 66228, 423128, 206370, 503751, 183277, 3729, 748722, 458365, 718350, 809876]
    Assuming 1802716 mutations distributed as [132233, 20332, 87230, 385685, 72716, 10145, 14385, 236424, 41658, 7514, 49565, 9365, 16946, 79753, 63006, 290888, 100064, 184807]
    Assuming 11662516 mutations distributed as [12455, 414439, 767475, 934808, 736428, 939007, 246830, 519188, 757097, 928630, 677821, 618899, 252669, 861058, 786052, 658594, 730914, 820152]
    Assuming 11005910 mutations distributed as [837577, 332454, 572163, 775665, 566672, 437871, 773472, 501226, 177055, 58376, 591301, 657162, 854754, 963443, 972486, 445721, 661734, 826778]
    Assuming 12232306 mutations distributed as [770699, 675669, 955080, 795936, 426854, 497480, 799802, 224632, 957329, 787178, 913988, 846753, 592952, 782939, 409642, 317902, 740746, 736725]
    Assuming 12916371 mutations distributed as [988943, 977398, 653303, 647377, 274853, 710246, 560256, 498232, 703598, 848299, 476209, 998447, 670819, 514877, 792272, 972655, 680498, 948089]
    Assuming 12634971 mutations distributed as [209040, 661397, 744212, 797773, 668856, 663388, 908171, 618883, 924496, 970015, 972647, 588142, 520055, 532705, 423537, 472721, 974141, 984792]
    Searching for 18-partition of 7054608 such that each element is <= 1000000
    Assuming 7054608 mutations distributed as [196398, 151859, 399798, 185880, 687680, 126454, 375699, 130844, 410325, 556694, 350302, 712528, 97215, 712777, 974682, 22661, 39064, 923748]
*/
