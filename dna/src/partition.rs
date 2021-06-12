use super::polynomial::Polynomial;
use cached::proc_macro::cached;
use num::{BigInt, BigRational, BigUint, Num, One};
use num::{ToPrimitive, Zero};
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

#[cached]
fn triangle_distribution(lower_bound: usize, element_bound: usize) -> DiscreteDistribution {
    let mut weights = Vec::with_capacity(element_bound);
    for i in 0..=lower_bound {
        weights.push(element_bound as f64 - (lower_bound as f64 / 2. - i as f64).abs());
    }
    weights
}

/// Generates a random `size`-partition of a number <= `total_bound` such that all elements are <= `element_bound`
fn random_partition_approximate_distribution<Rng: RngCore>(
    size: usize,
    element_bound: usize,
    total_bound: usize,
    rng: &mut Rng,
) -> (usize, Vec<usize>) {
    let mut partition = Vec::with_capacity(size);
    let mut current_sum = 0;

    for _ in 0..size {
        // current_sum + r <= t_max * n_simulataneous_mutations
        // <=> r <= t_max * n_simultaneous_mutations - current_sum
        let lower_bound = std::cmp::min(element_bound, total_bound - current_sum);
        let weights = triangle_distribution(lower_bound, element_bound);
        let distribution = WeightedIndex::new(&weights).unwrap();
        let r = distribution.sample(rng);
        current_sum += r;
        partition.push(r);
    }
    partition.shuffle(rng);
    let x = partition.iter().sum::<usize>();
    assert!(partition.iter().sum::<usize>() <= total_bound);
    (x, partition)
}

#[cached]
pub fn partition_generating_polynomial(element_bound: usize, size: usize) -> Polynomial<BigUint> {
    Polynomial::new(vec![BigUint::one(); element_bound + 1]).powi(size)
}

pub fn partition_generating_polynomials<F: Num + One + Clone + std::fmt::Display>(
    element_bound: usize,
    size: usize,
) -> Vec<Polynomial<F>> {
    let mut v = Vec::with_capacity(size);
    let mut p = Polynomial::new(vec![F::one(); element_bound + 1]);
    for i in 0..=size {
        //println!("p^{} = {}", i, p);
        v.push(p.clone());
        p *= &p.clone();
    }
    v
}

pub type DiscreteDistribution = Vec<f64>;

#[cached]
pub fn distributions(element_bound: usize, size: usize) -> Vec<DiscreteDistribution> {
    let mut weights = Vec::with_capacity(size);
    for poly in partition_generating_polynomials::<BigUint>(element_bound, size).into_iter() {
        let s = poly
            .coeffs
            .clone()
            .into_iter()
            .fold(BigUint::zero(), std::ops::Add::add);
        let weight: Vec<f64> = poly
            .coeffs
            .into_iter()
            .map(|n| BigRational::new_raw(BigInt::from(n), BigInt::from(s.clone())).reduced())
            .map(|rational| {
                let t = rational.trunc().to_u128().unwrap() as f64;
                let f = rational.fract();
                let ff = match (f.numer().to_u128(), f.denom().to_u128()) {
                    (Some(n), Some(d)) => n as f64 / d as f64,
                    x => {
                        //println!("{} / {}", f.numer(), f.denom());
                        10.0e-19
                    }
                };
                t + ff
            })
            .collect();
        weights.push(weight);
    }
    weights
}

fn random_partition_exact_distribution<Rng: RngCore>(
    size: usize,
    element_bound: usize,
    total_bound: usize,
    rng: &mut Rng,
) -> (usize, Vec<usize>) {
    let mut partition = Vec::with_capacity(size);
    let mut remainder = rng.gen_range(1..=total_bound);
    let weights = distributions(element_bound, size);
    for k in 0..size {
        let remaining_size = size - k;
        let distribution = &weights[remaining_size];
        let r = match WeightedIndex::new(&distribution[..=std::cmp::min(remainder, element_bound)])
        {
            Ok(d) => d.sample(rng),
            // if all probabilities in our range are virtually zero we just assume
            // a uniform distribution on our interval
            Err(_) => rng.gen_range(0..=remaining_size),
        };
        remainder = remainder.saturating_sub(r);
        partition.push(r);
    }
    partition.shuffle(rng);
    (partition.iter().sum(), partition)
}

pub fn random_partition<Rng: RngCore>(
    size: usize,
    element_bound: usize,
    total_bound: usize,
    rng: &mut Rng,
) -> (usize, Vec<usize>) {
    if element_bound <= 50 {
        random_partition_exact_distribution(size, element_bound, total_bound, rng)
    } else {
        random_partition_approximate_distribution(size, element_bound, total_bound, rng)
    }
}
