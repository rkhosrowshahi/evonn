

def distribution_based_strategy_init(key: jax.random.PRNGKey, strategy: str, x0: np.ndarray, steps: int, args: argparse.Namespace) -> None:
    std_init = args.es_std
    print(f"Total number of steps: {steps}")
            
    if strategy == 'CMA_ES':
        alpha = 1
        d = len(x0)
        base = 4 + np.floor(3 * np.log(max(1, d)))
        popsize = max(2, int(np.ceil(alpha * base)))
        args.popsize = popsize
        es = distribution_based_algorithms[strategy](
            population_size=popsize, 
            solution=x0,
        )
        es_params = es.default_params.replace(
            std_init=std_init,
            std_min=1e-6, 
            std_max=1e3
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'Sep_CMA_ES':
        alpha = 1
        d = len(x0)
        base = 4 + np.floor(3 * np.log(max(1, d)))
        popsize = max(2, int(np.ceil(alpha * base)))
        args.popsize = popsize
        es = distribution_based_algorithms[strategy](
            population_size=popsize, 
            solution=x0,
        )
        es_params = es.default_params.replace(
            std_init=std_init,
            std_min=1e-6, 
            std_max=1e3
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'SV_CMA_ES':
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize//5, 
            num_populations=5,
            solution=x0,
        )
        es_params = es.default_params.replace(std_init=std_init, std_min=1e-6, std_max=1e1)
        means = np.random.normal(x0, 0, (5, x0.shape[0]))
        es_state = es.init(key=key, means=means, params=es_params)
    elif strategy == 'SimpleES':
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=args.es_lr,
            boundaries_and_scales={
                steps // 160: args.es_lr,  # multiply by 0.1 at step 160
                steps // 180: args.es_lr,  # multiply by 0.1 again at step 180
            }
        )
        optimizer = None
        if args.es_optimizer == 'sgd':
            optimizer = optax.sgd(learning_rate=lr_schedule)
        elif args.es_optimizer == 'adam':
            optimizer = optax.adam(learning_rate=lr_schedule)
        else:
            raise ValueError(f"Invalid optimizer: {args.es_optimizer}")
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            optimizer=optax.sgd(learning_rate=lr_schedule),
        )
        es_params = es.default_params.replace(
            std_init=std_init,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'Open_ES':
        # For SGD optimizer
        # lr_schedule = optax.cosine_decay_schedule(
        #     init_value=1e-3,
        #     decay_steps=steps,
        #     alpha=1e-2,
        # )
        # lr_schedule = optax.piecewise_constant_schedule(
        #     init_value=args.es_lr,
        #     boundaries_and_scales={
        #         steps // 160: 0.1,  # multiply by 0.1 at step 160
        #         steps // 180: 0.1,  # multiply by 0.1 again at step 180
        #     }
        # )
        
        # lr_schedule = optax.piecewise_constant_schedule(
        #     init_value=args.es_lr,
        #     boundaries_and_scales={
        #         steps // 160: args.es_lr,  # multiply by 0.1 at step 160
        #         steps // 180: args.es_lr,  # multiply by 0.1 again at step 180
        #     }
        # )
        lr_schedule = optax.constant_schedule(args.es_lr)
        std_schedule = optax.cosine_decay_schedule(
            init_value=std_init,
            decay_steps=steps,
            alpha=1e-2,
        )
        optimizer = None
        if args.es_optimizer == 'sgd':
            lr_schedule = optax.cosine_decay_schedule(
                init_value=args.es_lr,
                decay_steps=steps,
                alpha=1e-6,
            )
            optimizer = optax.sgd(learning_rate=lr_schedule)
        elif args.es_optimizer == 'adam':
            print(f"Using Adam optimizer with learning rate: {args.es_lr}")
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=args.es_lr,
                boundaries_and_scales={
                    int(steps * 0.5): 0.1,  # multiply by 0.1 at step 160
                    int(steps * 0.8): 0.1,  # multiply by 0.1 again at step 180
                }
            )
            print(f"Piecewise constant schedule with steps {steps} at {int(steps * 0.5)}, {lr_schedule(int(steps * 0.5))} and {int(steps * 0.8)}, {lr_schedule(int(steps * 0.8))}")
            optimizer = optax.adam(learning_rate=lr_schedule)
        elif args.es_optimizer == 'adamw':
            optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=args.wd)
        else:
            raise ValueError(f"Invalid optimizer: {args.es_optimizer}")
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            optimizer=optimizer,
            std_schedule=std_schedule,
            use_antithetic_sampling=False,
        )
        es_params = es.default_params
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'SV_Open_ES':
        # For SGD optimizer
        lr_schedule = optax.cosine_decay_schedule(
            init_value=1e-3,
            decay_steps=steps,
            alpha=1e-2,
        )
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=args.es_lr,
            boundaries_and_scales={
                steps // 160: args.es_lr,  # multiply by 0.1 at step 160
                steps // 180: args.es_lr,  # multiply by 0.1 again at step 180
            }
        )
        std_schedule = optax.cosine_decay_schedule(
            init_value=std_init,
            decay_steps=steps,
            alpha=0.0,
        )
        optimizer = None
        if args.es_optimizer == 'sgd':
            optimizer = optax.sgd(learning_rate=lr_schedule)
        elif args.es_optimizer == 'adam':
            optimizer = optax.adam(learning_rate=lr_schedule)
        else:
            raise ValueError(f"Invalid optimizer: {args.es_optimizer}")
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize//5, 
            num_populations=5,
            solution=x0,
            optimizer=optimizer,
            std_schedule=std_schedule,
            use_antithetic_sampling=True,
        )
        es_params = es.default_params
        es_state = es.init(key=key, means=np.random.normal(x0, 0.1, (5, x0.shape[0])), params=es_params)
    elif strategy == 'xNES':
        # For SGD optimizer
        lr_schedule = optax.cosine_decay_schedule(
            init_value=1e-3,
            decay_steps=steps,
            alpha=1e-2,
        )
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=args.es_lr,
            boundaries_and_scales={
                steps // 160: 0.1,  # multiply by 0.1 at step 160
                steps // 180: 0.1,  # multiply by 0.1 again at step 180
            }
        )
        optimizer = None
        if args.es_optimizer == 'sgd':
            optimizer = optax.sgd(learning_rate=lr_schedule)
        elif args.es_optimizer == 'adam':
            optimizer = optax.adam(learning_rate=lr_schedule)
        else:
            raise ValueError(f"Invalid optimizer: {args.es_optimizer}")
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            optimizer=optimizer,
        )
        es_params = es.default_params
        es_params = es_params.replace(
            std_init=std_init,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)

    elif strategy == 'EvoTF_ES':
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0
        )
        es_params = es.default_params
        es_params = es_params.replace(
            std_init=std_init,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'LES':
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0
        )
        es_params = es.default_params
        es_params = es_params.replace(
            std_init=std_init,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'PGPE':
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0
        )
        es_params = es.default_params
        es_params = es_params.replace(
            std_init=std_init,
        )
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'iAMaLGaM_Full':
        std_schedule = optax.cosine_decay_schedule(
            init_value=args.es_std,
            decay_steps=steps,
            alpha=1e-2,
        )
        args.popsize = int(np.floor(10 * np.power(len(x0), 0.5)))
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            std_schedule=std_schedule,
        )
        es_params = es.default_params
        es_state = es.init(key=key, mean=x0, params=es_params)
    elif strategy == 'NoiseReuseES':
        std_schedule = optax.cosine_decay_schedule(
            init_value=args.es_std,
            decay_steps=steps,
            alpha=1e-2,
        )
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=args.es_lr,
            boundaries_and_scales={
                steps // 160: 0.1,  # multiply by 0.1 at step 160
                steps // 180: 0.1,  # multiply by 0.1 again at step 180
            }
        )
        if args.es_optimizer == 'sgd':
            optimizer = optax.sgd(learning_rate=lr_schedule)
        elif args.es_optimizer == 'adam':
            optimizer = optax.adam(learning_rate=lr_schedule)
        es = distribution_based_algorithms[strategy](
            population_size=args.popsize, 
            solution=x0,
            std_schedule=std_schedule,
            optimizer=optimizer,
        )
        es_params = es.default_params
        es_state = es.init(key=key, mean=x0, params=es_params)
    print(f"ES parameters: {es_params}")
    return es, es_params, es_state


def population_based_strategy_init(strategy: str, args: argparse.Namespace, x0: np.ndarray, steps: int) -> None:
    if strategy == 'DE':
        es = population_based_algorithms['DifferentialEvolution'](
            population_size=args.popsize, 
            solution=x0,
            num_diff=1,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            elitism=False,
            differential_weight=args.de_mr,
            crossover_rate=args.de_cr,
        )
    elif strategy == 'PSO':
        es = population_based_algorithms['PSO'](
            population_size=args.popsize, 
            solution=x0,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            inertia_coeff=args.pso_w,           # Balanced exploration/exploitation
            cognitive_coeff=args.pso_c1,         # Enhanced personal learning
            social_coeff=args.pso_c2,          # Enhanced global learning
        )
    elif strategy == 'DiffusionEvolution':
        es = population_based_algorithms['DiffusionEvolution'](
            population_size=args.popsize, 
            solution=x0,
            num_generations=steps,
            num_latent_dims=128,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params
    elif strategy == 'GA':
        std_schedule = optax.cosine_decay_schedule(
            init_value=args.ga_std,
            decay_steps=steps,
            alpha=1e-2,
        )
        es = population_based_algorithms['SimpleGA'](
            population_size=args.popsize, 
            solution=x0,
            std_schedule=std_schedule,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            crossover_rate=args.ga_cr,
        )
    elif strategy == 'LGA':
        es = population_based_algorithms['LGA'](
            population_size=args.popsize, 
            solution=x0,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params.replace(
            crossover_rate=args.ga_cr,
            std_init=1.0,
        )
    elif strategy == 'GESMR_GA':
        es = population_based_algorithms['GESMR_GA'](
            population_size=args.popsize, 
            solution=x0,
            num_groups=2,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params
    elif strategy == 'MR15_GA':
        es = population_based_algorithms['MR15_GA'](
            population_size=args.popsize, 
            solution=x0,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params
    elif strategy == 'SAMR_GA':
        es = population_based_algorithms['SAMR_GA'](
            population_size=args.popsize, 
            solution=x0,
            # fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        )
        es_params = es.default_params
    return es, es_params


STRATEGY_TYPES = {
    'cma_es': 'ES',
    'sep_cma_es': 'ES',
    'sv_cma_es': 'ES',
    'simplees': 'ES',
    'open_es': 'ES',
    'sv_open_es': 'ES',
    'xnes': 'ES',
    'de': 'EA',
    'pso': 'EA',
    'diffusionevolution': 'EA',
    'ga': 'EA',
    'lga': 'EA',
    'gesmr_ga': 'EA',
    'mr15_ga': 'EA',
    'samr_ga': 'EA',
    'evotf_es': 'ES',
    'les': 'ES',
    'pgpe': 'ES',
    'iamalgam_full': 'ES',
    'noisereusees': 'ES',
}
