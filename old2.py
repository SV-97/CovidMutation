def sample_markov(start: List[Nucleobase], alpha: float, beta: float, t_max, n_samples, seed=0):
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
    print(total_end_distribution)

    base = np.array((A, C, G, T))
    rng = np.random.default_rng(seed)
    samples = [rng.choice(base, n_samples, p=distribution)
               for distribution in total_end_distribution]
    """
    def sample_distribution(dist):
        # return rng.choice(base, p=dist)
        return random.choices(base, weights=dist, k=1)[0]

    # take n_samples samples from the end distribution
    samples = np.array([np.apply_along_axis(
        sample_distribution, axis=1, arr=total_end_distribution) for _ in range(n_samples)])
    """

    base = [A, C, G, T]
    base_arr = np.array(n_samples * [base])

    samples = rng.choice(base_arr, size=(n_samples, ),
                         p=total_end_distribution.flatten())
    exit()

    return samples
