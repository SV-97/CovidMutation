use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::RngCore;

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
pub fn random_partition<Rng: RngCore>(n: usize, m: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    if k * m < n {
        panic!(
            "Can't form a {}-partition of {} where each element is <= {}",
            m, n, k
        );
    }
    let mut partition = Vec::with_capacity(m);
    let mut remainder = n;
    println!(
        "Searching for {}-partition of {} such that each element is <= {}",
        m, n, k
    );

    // We first generate the probability distribution of our random partition
    let mut weights = Vec::with_capacity(n + 1);
    for i in 0..=std::cmp::min(k, remainder) {
        weights.push(bounded_k_partition_ratio(i, k, m, remainder));
    }
    let distribution = WeightedIndex::new(&weights).unwrap();
    loop {
        remainder = n;
        partition.clear();
        for i in 1..m {
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
        }
    }
    // partition.shuffle(rng);
    partition
}
