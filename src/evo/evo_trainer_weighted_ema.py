import argparse
import itertools
import os
import numpy as np
import torch
import wandb
from src.evo.schedulers import *
from src.models import get_model
from src.evo.adapters import adapter_checkpoint_dict, get_adapter
from src.utils import *
from pymoo.algorithms.soo.nonconvex.de import DE, Variant as DEVariant
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.operators.control import EvolutionaryParameterControl, NoParameterControl


def _param_l2(model: torch.nn.Module) -> torch.Tensor:
    v = params_to_vector(model.parameters())
    return v.pow(2).sum().item()


def _de_control_str_from_individual(ind) -> str:
    """Format pymoo EvolutionaryParameterControl fields (ParameterControl.*) for printing."""
    if ind is None:
        return ""
    parts = []
    f = ind.get("ParameterControl.F")
    if f is not None:
        parts.append(f"F={float(np.asarray(f).ravel()[0]):.4f}")
    cr = ind.get("ParameterControl.CR")
    if cr is not None:
        parts.append(f"CR={float(np.asarray(cr).ravel()[0]):.4f}")
    jitter = ind.get("ParameterControl.jitter")
    if jitter is not None:
        parts.append(f"jitter={bool(np.asarray(jitter).ravel()[0])}")
    n_diffs = ind.get("ParameterControl.n_diffs")
    if n_diffs is not None:
        parts.append(f"n_diffs={int(np.asarray(n_diffs).ravel()[0])}")
    selection = ind.get("ParameterControl.selection")
    if selection is not None:
        s = selection if isinstance(selection, str) else str(np.asarray(selection).ravel()[0])
        parts.append(f"selection={s}")
    cross = ind.get("ParameterControl.crossover")
    if cross is not None:
        c = cross if isinstance(cross, str) else str(np.asarray(cross).ravel()[0])
        parts.append(f"crossover={c}")
    return "  ".join(parts)


def main(args):
    l2_coef = 0.0 if args.l2_reg_wd is None else float(args.l2_reg_wd)
    # Seed first: DE uses pymoo's RNG, but batch order + CUDA ops must be fixed
    # before any CUDA init / wandb, or runs diverge with the same --seed.
    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.disable_wandb:
        # Still init so wandb.log() is a no-op; do not set WANDB_DISABLED (that blocks all API use).
        wandb.init(mode="disabled")
    else:
        wb_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "group": args.wandb_group,
            "name": args.wandb_name,
            "config": vars(args),
        }
        wandb.init(**wb_kwargs)

    # Check CUDA memory and provide information
    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        torch.cuda.empty_cache()  # Clear any existing cache
    else:
        print("Using CPU device")
        
    # Create save directory
    save_path = os.path.join(args.save_path)
    os.makedirs(save_path, exist_ok=True)
    print(f"Created save directory: {save_path}")
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

    print(f"Starting training with arguments: {args}")
    
    # Load data and model
    train_loader, val_loader, test_loader, num_classes, input_size = create_dataset(args)

    model = get_model(model_name=args.arch, input_size=input_size, num_classes=num_classes, device=device)
    theta_0 = params_to_vector(model.parameters())
    num_weights = len(theta_0)
    
    # Setup loss function
    criterion = create_criterion(args=args, num_classes=num_classes)
    
    param_adapter = get_adapter(args.adapter, model=model, args=args)
    
    num_dims = param_adapter.num_dims

    problem = Problem(
        n_var=num_dims,
        n_obj=1,
        n_constr=0,
        xl=args.pop_init_lb,
        xu=args.pop_init_ub,
    )

    pop_init = None
    if args.pop_init is not None:
        if args.pop_init == "normal":
            pop_init = rng.normal(0.0, args.pop_init_std, size=(args.pop_size, num_dims))
        elif args.pop_init == "uniform":
            pop_init = rng.uniform(args.pop_init_lb, args.pop_init_ub, size=(args.pop_size, num_dims))
        elif args.pop_init == "gram_schmidt_orthogonal":
            Z = rng.normal(0.0, 1.0, size=(args.pop_size, num_dims))
            Q, R = np.linalg.qr(Z.T)
            norms = np.linalg.norm(Q, axis=0)
            Q = (Q * norms).T
            pop_init = Q
        elif args.pop_init == "lhs":
            pop_init = LHS().do(problem, n_samples=args.pop_size, random_state=rng)
            pop_init = pop_init.get("X")
        else:
            raise ValueError(f"Invalid population initialization: {args.pop_init}")

    optimizer = None
    if args.optimizer == "de":

        de_param_control = EvolutionaryParameterControl if args.de_control else NoParameterControl
        de_variant = DEVariant(
            selection=getattr(args, 'de_selection', 'rand'),
            n_diffs=getattr(args, 'de_num_diffs', 1),
            F=getattr(args, 'de_mut_rate', 0.5),
            CR=getattr(args, 'de_cr_rate', 0.9),
            jitter=getattr(args, 'de_jitter', True),
            prob_mut=getattr(args, 'de_mut_prob', 1.0),
            control=de_param_control,
        )

        optimizer = DE(
            pop_size=args.pop_size,
            variant=de_variant,
            sampling=LHS(),
            seed=int(args.seed),
            verbose=False,
        )

    elif args.optimizer == "pso":

        optimizer = PSO(
            pop_size=args.pop_size,
            w=getattr(args, 'pso_w', 0.9),
            c1=getattr(args, 'pso_c1', 2.0),
            c2=getattr(args, 'pso_c2', 2.0),
            initial_velocity=getattr(args, 'pso_initial_velocity', 'random'),
            max_velocity_rate=getattr(args, 'pso_max_velocity_rate', 0.20),
            pertube_best=getattr(args, 'pso_pertube_best', False),
            adaptive=getattr(args, 'pso_adaptive', False),
            seed=int(args.seed),
            verbose=True,
        )

    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}, choices: de, pso")

    optimizer.setup(problem, termination=NoTermination(), verbose=True)
    
    step = 0
    fe_count = 0
    best_test_metrics: dict | None = None
    momentum = 0.0

    train_loader_iterator = itertools.cycle(train_loader)

    # Full-space evolutionary optimization (DE)

    for iteration in range(1, args.num_iterations + 1):
        if args.num_fe is not None and fe_count >= args.num_fe:
            print(
                f"Stopping: function evaluation budget reached (num_fe={args.num_fe}, fe_count={fe_count})."
            )
            break
        step += 1

        curr_batch = next(train_loader_iterator)

        # print(f"Iteration {iteration} - Current batch labels: {curr_batch[1][:10]}")

        # if optimizer.pop is not None and iteration > 1:
            # pop_F_weighted = (1/f_i)*f_i was ~1 per member -> sum ~ pop_size (wrong).
            # Weights w_i ∝ 1/f_i (normalized): sum_i w_i f_i = n / sum_j(1/f_j) (harmonic mean).
            # pop_F = np.asarray(optimizer.pop.get("F"), dtype=np.float64).ravel()
            # momentum = pop_F.mean()
            # inv = 1.0 / (pop_F + 1e-12)
            # pop_fitness_hmean = float(pop_F.size / np.sum(inv))
            # momentum = (
            #     args.fitness_ema_beta * momentum
            #     + (1.0 - args.fitness_ema_beta) * pop_fitness_hmean
            # )

            # IDEA 1:
            # momentum = (
            #     args.fitness_ema_beta * momentum
            #     + (1.0 - args.fitness_ema_beta) * pop_fitness_hmean
            # )
            # IDEA 2: 
            # momentum = pop_fitness_hmean
        #     pop_X = optimizer.pop.get("X")
        #     pop_F = optimizer.pop.get("F")
        #     pop_loss = optimizer.pop.get("loss")

        #     # pop_F_sorted_idx = np.argsort(pop_F)
        #     # momentum = None
        #     # IDEA 1: Weighted EMA based on fitness
        #     # weights = 1.0 / (pop_F + 1e-12)
        #     # pop_F_weighted = weights * pop_F
        #     # momentum = pop_F_weighted.sum()
        #     # IDEA 2: No UPDATE! -> don't reevaluate previous population.
        #     # IDEA 3: use same solution's fitness as momentum: pop_F[i]

        #     for i in range(len(pop_X)):
        #         param_adapter.apply(
        #                 theta_0,
        #                 pop_X[i],
        #                 alpha=args.alpha,
        #                 device=device,
        #             )
        #         loss = 0
        #         F = 0
        #         l2 = 0
        #         with torch.no_grad():
        #             x, y = curr_batch
        #             x, y = x.to(device), y.to(device)
        #             output = param_adapter.model(x)
        #             loss += criterion(output, y)
        #             l2 += _param_l2(param_adapter.model)
        #             F += (loss + l2_coef * l2)
                
        #         pop_loss[i, 0] = loss
        #         pop_F[i, 0] = args.fitness_momentum_mu * momentum + F
        #         fe_count += 1

        #     # TODO: Update population with moving average of fitnesses; currently not implemented.
        #     optimizer.pop.set("F", pop_F)
        #     optimizer.pop.set("loss", pop_loss)

        if optimizer.pop is not None and iteration > 1:
            pop_X = optimizer.pop.get("X")
            pop_F = optimizer.pop.get("F")
            pop_loss = optimizer.pop.get("loss")

            for i in range(len(pop_X)):
                param_adapter.apply(
                        theta_0,
                        pop_X[i],
                        alpha=args.alpha,
                        device=device,
                    )
                loss = 0
                F = 0
                l2 = 0
                with torch.no_grad():
                    x, y = curr_batch
                    x, y = x.to(device), y.to(device)
                    output = param_adapter.model(x)
                    loss += criterion(output, y)
                    l2 += _param_l2(param_adapter.model)
                    F += (loss + l2_coef * l2)
                
                pop_loss[i, 0] = loss
                pop_F[i, 0] = args.fitness_ema_beta * momentum + (1 - args.fitness_ema_beta) * F
                pop_F[i, 0] = pop_F[i, 0] / (1 - (args.fitness_ema_beta)**iteration)
                fe_count += 1

            # TODO: Update population with moving average of fitnesses; currently not implemented.
            optimizer.pop.set("F", pop_F)
            optimizer.pop.set("loss", pop_loss)
        
        offspring = optimizer.ask()
        if iteration == 1 and pop_init is not None:
            print(f"Setting initial population: {pop_init.shape}")
            print(f"Initial population min: {pop_init.min()}, max: {pop_init.max()}")
            offspring.set("X", pop_init.copy())

        offspring_X = offspring.get("X")
        offspring_loss = np.zeros((len(offspring_X), 1))
        offspring_F = np.zeros((len(offspring_X), 1))
        offspring_l2 = np.zeros((len(offspring_X), 1))
        for i in range(len(offspring_X)):
            param_adapter.apply(
                theta_0,
                offspring_X[i],
                alpha=args.alpha,
                device=device,
            )
            loss = 0
            F = 0
            l2 = 0
            with torch.no_grad():
                x, y = curr_batch
                x, y = x.to(device), y.to(device)
                output = param_adapter.model(x)
                loss += criterion(output, y)
                l2 += _param_l2(param_adapter.model)
                F += (loss + l2_coef * l2)
            
            offspring_loss[i, 0] = loss
            offspring_F[i, 0] = F
            offspring_l2[i, 0] = l2
            fe_count += 1

        offspring_F[:, 0] = 0.1 * momentum + 0.9 * offspring_F[:, 0]
        offspring_F[:, 0] = offspring_F[:, 0] / (1 - 0.9**iteration)

        static = StaticProblem(problem, F=offspring_F, loss=offspring_loss, l2=offspring_l2)
        Evaluator().eval(static, offspring)
        optimizer.tell(infills=offspring)

        if iteration >= 1:
            pop_F = np.asarray(optimizer.pop.get("F"), dtype=np.float64).ravel()
            topk_pop_F = np.argsort(pop_F)[:10]
            # momentum = 0.1 * momentum + 0.9 * pop_F[topk_pop_F].mean()
            # momentum = momentum / (1 - 0.9**iteration) # TODO: check if this works!
            momentum = pop_F[topk_pop_F].mean()

        result = optimizer.result()
        best_idx = np.argmin(result.F)
        best_X = result.opt.get("X")[best_idx]
        best_l2 = result.opt.get("l2")[best_idx, 0]
        best_loss = result.opt.get("loss")[best_idx, 0]
        best_F = result.opt.get("F")[best_idx, 0]
        # DE strategy params (F, CR, …) are stored on population members when de_control is on.
        best_pop_idx = int(np.argmin(np.asarray(result.pop.get("F")).reshape(-1)))
        # param_adapter.apply(theta_0, best_X, alpha=args.alpha, device=device)
        # best_loss = 0
        # best_F = 0
        # with torch.no_grad():
        #     x, y = curr_batch
        #     x, y = x.to(device), y.to(device)
        #     output = param_adapter.model(x)
        #     best_loss = criterion(output, y)
        #     v_b = _param_l2(param_adapter.model)
        #     best_F = (best_loss + l2_coef * v_b).item()
        # best_l2 = v_b.item()
        if args.de_control:
            s = _de_control_str_from_individual(result.pop[best_pop_idx])
            if s:
                print(f"[iter {iteration}] de_control (best pop member): {s}")

        wandb.log(
            {
                "evo/best/f": best_F,
                "evo/best/loss": best_loss,
                "evo/best/l2_reg": best_l2,
                "evo/best/d_min": best_X.min(),
                "evo/best/d_max": best_X.max(),
                "evo/best/d_mean": best_X.mean(),
                "evo/best/d_std": best_X.std(),
            },
            step=step,
        )

        top10_idx = np.argsort(result.pop.get("F"))[:10]
        center_X = np.mean(result.pop.get("X")[top10_idx], axis=0)  
        param_adapter.apply(theta_0, center_X, alpha=args.alpha, device=device)
        center_loss = 0
        center_F = 0
        center_l2 = 0
        with torch.no_grad():
            x, y = curr_batch
            x, y = x.to(device), y.to(device)
            output = param_adapter.model(x)
            center_loss = criterion(output, y)
            center_l2 = _param_l2(param_adapter.model)
            center_F = (center_loss + l2_coef * center_l2)
        wandb.log(
            {
                "evo/center/f": center_F,
                "evo/center/loss": center_loss,
                "evo/center/l2_reg": center_l2,
                "evo/center/d_min": center_X.min(),
                "evo/center/d_max": center_X.max(),
                "evo/center/d_mean": center_X.mean(),
                "evo/center/d_std": center_X.std(),
            },
            step=step,
        )

        if center_F > best_F:
            param_adapter.apply(theta_0, best_X, alpha=args.alpha, device=device)

        if args.test_interval is not None and iteration % args.test_interval == 0:
            test_metrics = evaluate_model_on_test(model=param_adapter.model, data_loader=test_loader, device=device)

            log_dict = {"test/loss": test_metrics["loss"], "test/top1": test_metrics["top1"], "test/top5": test_metrics["top5"], "test/f1": test_metrics["f1"]}
            if best_test_metrics is None or test_metrics["top1"] > best_test_metrics["top1"]:
                best_test_metrics = {k: float(v) for k, v in test_metrics.items()}
                ckpt_path = os.path.join(save_path, "checkpoints", "best.pt")
                torch.save(
                    {
                        "iteration": iteration,
                        "step": step,
                        "adapter": args.adapter,
                        "model_state_dict": param_adapter.model.state_dict(),
                        "adapter_state": adapter_checkpoint_dict(param_adapter),
                        "test_metrics": dict(best_test_metrics),
                    },
                    ckpt_path,
                )
                print(
                    f"New best test (by top1) — {', '.join(f'{k}={v:.4f}' for k, v in best_test_metrics.items())} — saved to {ckpt_path}"
                )
                for k, v in best_test_metrics.items():
                    log_dict[f"test/best/{k}"] = v
            wandb.log(log_dict, step=step)
        
        wandb.log(
            {
                "evo/iteration": iteration,
                "evo/function_evaluation": fe_count,
            },
            step=step,
        )
    
    if not args.disable_wandb:
        ckpt_path = os.path.join(save_path, "checkpoints", "best.pt")
        log_checkpoint_wandb(
            ckpt_path,
            metadata={**best_test_metrics, "iteration": iteration, "step": step},
        )
    if not args.disable_wandb:
        wandb.finish()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    
    # ============================================================================
    # Model and Dataset Configuration
    # ============================================================================
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (None for random seed)')
    parser.add_argument('--arch', type=str, default='resnet32',
                       help='Neural network architecture to use (e.g., resnet32, vgg16)', required=True)
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to train on (e.g., cifar10, cifar100, mnist)', required=True)
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.0,
                       help='Validation split for training')
    parser.add_argument('--sampler', type=str, default=None,
                       help='Sampler for training')
    parser.add_argument('--criterion', type=str, default='ce', choices=['ce', 'cm', 'mse', 'f1', 'soft_f1', 'ce_sf1'],
                       help='Loss function: ce (cross-entropy), cm (confidence margin), mse (mean squared error), f1 (F1 score), soft_f1 (soft F1), or ce_sf1 (CE + soft F1)')
    
    # ============================================================================
    # Training Hyperparameters
    # ============================================================================
    parser.add_argument('--f1_temperature', '--temperature', type=float, default=None,
                       help='Temperature for soft F1 score error')
    parser.add_argument('--f1_beta', type=float, default=None,
                       help='Beta for soft F1 score error')
    parser.add_argument('--f1_learnable_temperature', type=bool, default=False,
                       help='Learnable temperature for soft F1 score error')
    parser.add_argument('--ce_weight', type=float, default=None,
                       help='Weight for CrossEntropy loss in combined loss (default: 0.5)')
    parser.add_argument('--f1_weight', type=float, default=None,
                       help='Weight for F1 loss in combined loss (default: 0.5)')
    parser.add_argument('--label_smoothing', type=float, default=None,
                       help='Label smoothing for soft F1 score error, use 0.1 for CIFAR-10 and 0.05 for CIFAR-100 datasets')

    # ============================================================================
    # Optimizer and Learning Rate Configuration
    # ============================================================================
    parser.add_argument('--bus_lr', "--learning_rate", type=float, default=None,
                       help='Initial learning rate')
    parser.add_argument('--bus_lr_scheduler', type=str, default=None, 
                       choices=['cosine', 'step', 'multi_step', 'constant'],
                       help='Learning rate scheduler type')
    parser.add_argument('--bus_lr_scheduler_step_size', type=int, default=None,
                       help='Step size for step scheduler')
    parser.add_argument('--bus_lr_scheduler_gamma', type=float, default=None,
                       help='Gamma (decay factor) for step/multi_step schedulers')
    parser.add_argument('--bus_lr_scheduler_milestones', type=str, default=None,
                       help='Milestones for multi_step lr scheduler (comma-separated)')
    parser.add_argument('--l2_reg_wd', type=float, default=None,
                       help='Weight decay (L2 regularization) coefficient')
    parser.add_argument('--momentum', type=float, default=None,
                       help='Momentum factor for SGD optimizer')
    
    # ============================================================================
    # Evolutionary Strategy Configuration
    # ============================================================================
    parser.add_argument('--optimizer', type=str, default=None,
                       help='Evolutionary optimizer to use (EA: PSO, etc. | ES: CMA_ES, SV_CMA_ES, SimpleES, Open_ES, SV_Open_ES, xNES)', required=True)
    # General evolutionary optimization configuration
    parser.add_argument('--pop_size', type=int, default=50, help='Population size')
    parser.add_argument('--num_iterations', type=int, default=None, help='Number of evolution iterations')
    parser.add_argument(
        '--num_fe',
        type=int,
        default=None,
        help='Max fitness evaluations (population re-eval + offspring per iteration); stop at the start of the next iteration when the counter has reached this (None disables).',
    )
    parser.add_argument('--pop_init', type=str, default=None, help='Initial distribution (normal or uniform)')
    parser.add_argument('--pop_init_lb', type=float, default=None, help='Lower bound for uniform pop_init')
    parser.add_argument('--pop_init_ub', type=float, default=None, help='Upper bound for uniform pop_init')
    parser.add_argument('--pop_init_std', type=float, default=None, help='Standard deviation for normal pop_init')
    # DE optimizer configuration
    parser.add_argument('--de_cr_rate', type=float, default=None,
                       help='Crossover rate for DE optimizer')
    parser.add_argument('--de_mut_rate', type=float, default=None,
                       help='Mutation rate for DE optimizer')
    parser.add_argument('--de_selection', type=str, default=None,
                       choices=['rand', 'best', 'rand-to-best', 'current-to-rand', 'current-to-best'],
                       help='Selection method for DE optimizer')
    parser.add_argument('--de_num_diffs', type=int, default=None,
                       help='Number of differences for DE optimizer')
    parser.add_argument('--de_jitter', action='store_true', default=False,
                       help='Jitter factor for DE optimizer')
    parser.add_argument('--de_mut_prob', type=float, default=1.0,
                       help='Mutation probability for DE optimizer')
    parser.add_argument(
        '--de_control', action='store_true', default=False,
        help='Use pymoo EvolutionaryParameterControl to adapt F/CR/jitter/selection (less bit-reproducible). Default: fixed DE parameters (NoParameterControl).',
    )
    # GA optimizer configuration
    parser.add_argument('--ga_std', type=float, default=None,
                       help='Standard deviation for GA noise generation (used when strategy=ga)')
    parser.add_argument('--ga_mut_prob', type=str, default=None)
    parser.add_argument('--ga_cr_prob', type=float, default=None)
    parser.add_argument('--ga_mut_eta', type=str, default=None)
    parser.add_argument('--ga_cr_eta', type=float, default=None)
    # PSO optimizer configuration
    parser.add_argument('--pso_w', type=float, default=None,
                       help='Inertia coefficient for PSO optimizer (default: 0.9)')
    parser.add_argument('--pso_c1', type=float, default=None,
                       help='Cognitive coefficient for PSO optimizer (default: 2.0)')
    parser.add_argument('--pso_c2', type=float, default=None,
                       help='Social coefficient for PSO optimizer (default: 2.0)')
    parser.add_argument('--pso_initial_velocity', type=str, default=None, choices=['random', 'zero'],
                       help='Initial velocity for PSO optimizer (default: random)')
    parser.add_argument('--pso_max_velocity_rate', type=float, default=None,
                       help='Max velocity rate for PSO optimizer (default: 0.20)')
    parser.add_argument('--pso_pertube_best', action='store_true', default=False,
                       help='Perturb the global best for PSO optimizer')
    parser.add_argument('--pso_adaptive', action='store_true', default=False,
                       help='Adaptive for PSO optimizer')
    # ES optimizer configuration
    parser.add_argument('--es_std', type=float, default=None,
                       help='Standard deviation for ES noise generation')
    parser.add_argument('--es_lr', type=float, default=None,
                       help='Learning rate for ES optimizer')
    parser.add_argument('--es_optimizer', type=str, default=None, choices=['sgd', 'adam', 'adamw', 'none'],
                       help='Mean (mu) vector optimizer (updater) in ES optimizer')
    parser.add_argument('--fitness_ema_beta', type=float, default=0.0,
                       help='Exponential moving average beta for fitness (default: 0.0 = replace new batch fitness fully)')
    
    # ============================================================================
    # Parameter Adapter Configuration
    # ============================================================================
    parser.add_argument('--adapter', type=str, default='full',
                       help='Ignored; full-weight DE only. Optional for shared YAML configs.')
    parser.add_argument('--adapter_k', type=str, default=None,
                       help='Number of random projections for random projection adapter')
    parser.add_argument('--adapter_rank', type=str, default=None,
                       choices=['value', 'index'],
                       help='Bin strategy for global uniform binning')
    
    # ============================================================================
    # Logging and Output Configuration
    # ============================================================================
    parser.add_argument('--save_path', type=str, default='logs/results',
                       help='Directory to save training results and checkpoints')
    
    # ============================================================================
    # Weights & Biases (wandb) Configuration
    # ============================================================================
    parser.add_argument('--wandb_project', type=str, default='evows',
                       help='Wandb project name for experiment tracking')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity (team/user) for experiment tracking')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (auto-generated if None)')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group for comparing runs (e.g. sweeps or experiment family)')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging completely')
    parser.add_argument('--note', type=str, default=None,
                       help='Additional note/description for the experiment run')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Scaling when applying adapter perturbations')
    parser.add_argument('--test_interval', type=int, default=None,
                       help='Evaluate on the test set every N iterations (None to disable)')

    args = parser.parse_args()
    main(args)

