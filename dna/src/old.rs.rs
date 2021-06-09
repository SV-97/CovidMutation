fn sample_markov<const N_SAMPLES: usize>(
    start: Sequence,
    alpha: f64,
    beta: f64,
    t_max: usize,
) -> [Sequence; N_SAMPLES] {
    let gamma = 1. - 2. * beta - alpha;
    let p_single = Mat::new([
        [gamma, beta, alpha, beta],
        [beta, gamma, beta, alpha],
        [alpha, beta, gamma, beta],
        [beta, alpha, beta, gamma],
    ]);
    let p_single_end = p_single.powi(t_max);
    let i = Mat::<4, 4>::identity();
    let start_distribution = {
        let mut m = Mat::<{ 6 * 3 }, 4>::zero();
        for (i, codon) in start.iter().enumerate() {
            let j = 3 * i;
            m[(j, codon.a() as usize)] = 1.0;
            m[(j + 1, codon.b() as usize)] = 1.0;
            m[(j + 2, codon.c() as usize)] = 1.0;
        }
        m
    };
    let total_end_distribution = start_distribution * p_single_end;
    // println!("{}", &total_end_distribution);
    let base = [Nucleobase::A, Nucleobase::C, Nucleobase::G, Nucleobase::T];
    let mut sample = [[Codon::default(); 6]; N_SAMPLES];
    let mut rng = thread_rng();
    for r in sample.iter_mut() {
        let mut seq = [Codon::default(); 6];
        let mut f = |nu| {
            *base
                .choose_weighted(&mut rng, |item| {
                    total_end_distribution[(nu, *item as usize)]
                })
                .unwrap()
        };
        for (j, s) in seq.iter_mut().enumerate() {
            let nu = j * 3;
            *s = Codon((f(nu), f(nu + 1), f(nu + 2)));
        }
        *r = seq;
    }
    sample
}

fn simulate_markov<const N_SAMPLES: usize, F: Fn(&Sequence) -> bool>(
    start: Sequence,
    end_predicate: F,
    alpha: f64,
    beta: f64,
    t_max: usize,
) -> f64 {
    let samples = sample_markov::<N_SAMPLES>(start, alpha, beta, t_max);
    let mut count = 0;
    for sample in &samples {
        if end_predicate(sample) {
            count += 1;
        }
    }
    (count as f64) / (N_SAMPLES as f64)
}

/// Generate random m-partition of n where each entry is <= k
/// To generate a normal m-partition of n with this set k=n.
/// Doesn't generate uniformly distsributed partitions
fn random_partition<Rng: RngCore>(n: usize, m: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    if k * m < n {
        panic!(
            "Can't form a {}-partition of {} where each element is <= {}",
            m, n, k
        );
    }
    let mut partition = Vec::with_capacity(m);
    let mut remainder = n;
    // We first generate the probability distribution of our random partition
    let mut distribution = Vec::with_capacity(n + 1);
    for i in 0..=k {
        distribution.push(bounded_k_partitions(i, k, m, n));
    }

    println!(
        "Distribution {:?} with sum {}",
        &distribution,
        distribution.iter().sum::<usize>()
    );
    let index = WeightedIndex::new(&distribution).unwrap();
    for i in 1..m {
        /*
        Let r be the new random entry. Then we have r <= k.
        We also know that the sum of the remaining entries has to equal Σ = remainder - r, but
        because each of the m-i remaining entries is <= k we also have Σ <= k (m-i).
        It follows that remainder-r <= k (m-i) <=> r >= k (i-m) + remainder.
        We rewrite this as remainder - k (m-i) to avoid underflows in i-m.
        This means that our current entry has to be in the range [remainder - k (m-i), k].
        To avoid underflows in case k (m - i) is > remainder we make our lower bound floor
        out at 0 and for cases where k > n we take an upper bound u <= n.
        */
        let lower_bound = remainder.saturating_sub(k * (m - i));
        let upper_bound = std::cmp::min(k, remainder);
        // generate random number in remaining range
        let r = rng.gen_range(lower_bound..=upper_bound);
        partition.push(r); // add that number to the partition
        remainder -= r;
    }
    partition.push(remainder);
    partition.shuffle(rng);
    partition
}

fn distribution test() {
    let h = [0; 10_000_000]
        .iter()
        .map(|_| random_partition(20, 4, 20, &mut master_rng))
        .fold(HashMap::<[usize; 4], usize>::new(), |mut h, p| {
            let mut p: [usize; 4] = p.try_into().unwrap();
            p.sort();
            let count = h.entry(p).or_insert(0);
            *count += 1;
            h
        });
    for (key, value) in h {
        println!("{:?} : {}", key, value);
    }
    

}