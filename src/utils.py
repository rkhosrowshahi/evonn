import argparse
import os
import random
import sys
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Sampler
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.population_based import population_based_algorithms
from evosax.core.fitness_shaping import *
from scipy.stats import norm
import wandb

# --------------------- Model Save/Load Functions ---------------------

def save_model(model: nn.Module, name: str, wandb_run=None) -> None:
    """Save model to Weights & Biases.
    
    Args:
        model: The PyTorch model to save
        name: Name to save the model as
        wandb_run: Weights & Biases run instance (optional)
    """
    if wandb_run is None:
        wandb_run = wandb
    
    if wandb_run.run is not None:
        save_path = os.path.join(wandb_run.run.dir, f'{name}.pt')
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to: {save_path}')
        
        # Also save as wandb artifact
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(save_path)
        wandb_run.log_artifact(artifact)

def log_checkpoint_wandb(
    ckpt_path: str,
    *,
    artifact_name: str = "best",
    artifact_type: str = "checkpoint",
    metadata: dict | None = None,
) -> None:
    """Register an on-disk checkpoint with the current wandb run as a versioned artifact.

    No-op if there is no active run, or the file is missing. Mirrors :func:`save_model` artifact use.
    """
    if not os.path.isfile(ckpt_path):
        return
    if wandb.run is None:
        return
    art = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata or {})
    art.add_file(ckpt_path, name="best.pt")
    wandb.log_artifact(art)

def load_model(name: str) -> Dict:
    """Load a saved model.
    
    Args:
        name: Path to the saved model
        
    Returns:
        The loaded model state dictionary
    """
    return torch.load(name, map_location=torch.device('cpu'))

# --------------------- Dataset Functions ---------------------

def get_balanced_indices(dataset, split_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get balanced indices for train/validation split using stratification.
    
    Args:
        dataset: PyTorch dataset
        split_size: Size of the validation split
        
    Returns:
        Tuple of (validation indices, training indices)
    """
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    indices = range(len(dataset))
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=split_size,
        stratify=all_labels,
        random_state=42,
        shuffle=True
    )
    
    return torch.tensor(val_idx), torch.tensor(train_idx)

# Helper function to create weighted sampler and loader
class BalancedBatchSampler(Sampler):
    """
    Sampler that yields balanced batches where each class has approximately equal representation.
    For CIFAR-10 with batch_size=256 and 10 classes: 25-26 samples per class per batch.
    """
    def __init__(self, dataset, batch_size: int, num_classes: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Get labels for all samples
        # Handle both Subset and full dataset
        if isinstance(dataset, Subset):
            self.labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        else:
            self.labels = np.array(dataset.targets)
        
        # Create indices for each class
        self.class_indices = {c: np.where(self.labels == c)[0] for c in range(num_classes)}
        
        # Calculate samples per class per batch
        self.base_samples_per_class = batch_size // num_classes
        self.extra_samples = batch_size % num_classes
        
        # Calculate total number of batches based on smallest class
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.base_samples_per_class
        
    def __iter__(self):
        # Shuffle indices for each class
        shuffled_class_indices = {
            c: np.random.permutation(indices).tolist() 
            for c, indices in self.class_indices.items()
        }
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Determine which classes get extra samples (rotate to distribute fairly)
            classes_with_extra = set((batch_idx + i) % self.num_classes for i in range(self.extra_samples))
            
            for class_id in range(self.num_classes):
                # Calculate how many samples from this class
                n_samples = self.base_samples_per_class
                if class_id in classes_with_extra:
                    n_samples += 1
                
                # Get samples for this class
                start_idx = batch_idx * self.base_samples_per_class
                end_idx = start_idx + n_samples
                
                class_batch = shuffled_class_indices[class_id][start_idx:end_idx]
                batch.extend(class_batch)
            
            # Shuffle the batch to mix classes
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.num_batches


def create_inverse_balanced_loader(train_dataset: Subset) -> WeightedRandomSampler:
        labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

def create_dataset(args) -> Tuple[Subset, Subset, Subset, int, int]:
    """Load and prepare dataset with train/val/test splits.
    
    Args:
        dataset: Name of dataset ('cifar100', 'cifar10', or 'mnist')
        validation_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
        
    Raises:
        ValueError: If dataset name is not supported
    """
    dataset = args.dataset
    batch_size = args.batch_size
    test_batch_size = batch_size
    if dataset == 'cifar100':
        num_classes = 100
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        dataset_class = torchvision.datasets.CIFAR100
        input_size = 32
    elif dataset == 'cifar10':
        num_classes = 10
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        dataset_class = torchvision.datasets.CIFAR10
        input_size = 32
    elif dataset == 'mnist':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_class = torchvision.datasets.MNIST
        transform_train = transform_test = transform
        input_size = 28
        test_batch_size = 10000
    elif dataset == 'fashion':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset_class = torchvision.datasets.FashionMNIST
        transform_train = transform_test = transform
        input_size = 28
        test_batch_size = 10000
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Load datasets
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    
    # Create validation split
    validation_split = args.val_split
    if validation_split > 0:
        val_indices, train_indices = get_balanced_indices(train_dataset, split_size=int(validation_split * len(train_dataset)))
        # Create subsets from original dataset
        val_dataset = Subset(train_dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        train_dataset = Subset(train_dataset, train_indices)
    else:
        val_loader = None
    # Create data loaders
    train_loader = None
    
    dataloader_gen: Optional[torch.Generator] = None
    s = getattr(args, "seed", None)
    if s is not None:
        dataloader_gen = torch.Generator()
        dataloader_gen.manual_seed(int(s))

    if args.sampler is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=dataloader_gen,
            pin_memory=False,
            num_workers=0
        )
    else:
        if args.sampler == 'inverse':
            sampler = create_inverse_balanced_loader(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=False,
                num_workers=0
            )
        elif args.sampler == 'balanced':
            batch_sampler = BalancedBatchSampler(train_dataset, batch_size, num_classes)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                pin_memory=False,
                num_workers=0
            )
        elif args.sampler == 'random':
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=dataloader_gen,
                pin_memory=False,
                num_workers=0
            )
        else:
            raise ValueError(f"Invalid sampler: {args.sampler}")
   
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader, num_classes, input_size

def create_balanced_dataset(dataset, num_classes: int, samples_per_class: int = None) -> Subset:
    """Create a balanced dataset with equal samples per class.
    
    Args:
        dataset: PyTorch dataset (can be a Subset or regular dataset)
        num_classes: Number of classes in the dataset
        samples_per_class: Number of samples per class. If None, uses minimum class count
        
    Returns:
        Subset: Balanced dataset with equal samples per class
    """
    # Get all labels from the dataset
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)
    
    # Group indices by class
    class_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Determine samples per class
    if samples_per_class is None:
        # Use the minimum class count to ensure balance
        samples_per_class = min(len(indices) for indices in class_indices.values())
    
    # Sample balanced indices
    balanced_indices = []
    for class_label in range(num_classes):
        if class_label in class_indices:
            class_idx = class_indices[class_label]
            # Randomly sample from this class
            np.random.shuffle(class_idx)
            selected_indices = class_idx[:samples_per_class]
            balanced_indices.extend(selected_indices)
    
    # Shuffle the final indices to mix classes
    np.random.shuffle(balanced_indices)
    
    return Subset(dataset, balanced_indices)

# --------------------- Training Utilities ---------------------

class WarmUpLR(_LRScheduler):
    """Warmup learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        total_iters: Total iterations for warmup phase
        last_epoch: The index of last epoch
    """
    def __init__(self, optimizer, total_iters: int, last_epoch: int = -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on warmup progress."""
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------- Model Evaluation Functions ---------------------

def evaluate_model_on_test(
    model: nn.Module,
    data_loader: DataLoader,
    train: bool = False,
    device: str = 'cuda'
) -> Tuple[float, float, float]:
    """Evaluate model cross entropy loss, accuracy and F1 score on a dataset.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch
        
    Returns:
        Tuple of (loss, accuracy, f1_score)
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    top1 = AverageMeter('Top1', ':6.4f')
    top5 = AverageMeter('Top5', ':6.4f')
    f1 = 0
    ce_loss = AverageMeter('CE Loss', ':6.4f')
    num_batches = len(data_loader)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            ce_loss.update(criterion(logits, targets).item())
            _, predicted = torch.max(logits.data, 1)
            
            # Collect predictions and labels for F1 score
            all_predictions.extend(predicted)
            all_targets.extend(targets)

            topk = accuracy(logits, targets, topk=(1,5))
            top1.update(topk[0].item(), targets.size(0))
            top5.update(topk[1].item(), targets.size(0))

            if train:
                break
    
    acc1 = top1.avg
    acc5 = top5.avg
    f1 = f1_score(y_true=torch.tensor(all_targets).cpu().numpy(), y_pred=torch.tensor(all_predictions).cpu().numpy(), average='macro')
    
    return {"loss": ce_loss.avg, "top1": acc1, "top5": acc5, "f1": f1}

def evaluate_model_acc(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    train: bool = False
) -> float:
    """Evaluate model accuracy on a dataset.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch and negate for minimization
        
    Returns:
        Classification accuracy as percentage
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if train:
                correct *= -1  # Negate for minimization in optimization
                break
                
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model_ce(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    train: bool = False
) -> float:
    """Evaluate model using cross entropy loss.
    
    Args:
        model: Neural network model
        data_loader: DataLoader containing the dataset
        device: Device to run evaluation on
        train: If True, only evaluate on one batch
        
    Returns:
        Average cross entropy loss
    """
    model.eval()
    model.to(device)
    loss = 0
    total_batch = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += nn.functional.cross_entropy(outputs, labels).item()
            total_batch += 1
            
            if train:
                break
                
    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21  # Handle numerical instability
        
    return loss

def evaluate_model_acc_single_batch(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    train: bool = False
) -> float:
    """Evaluate model accuracy on a single batch.
    
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
        
    Returns:
        Negative classification accuracy as percentage (for minimization)
    """
    model.eval()
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        
    return -100 * correct / total

def evaluate_model_f1score_single_batch(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    train: bool = False
) -> float:
    """Evaluate model F1 score on a single batch.
    
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
        
    Returns:
        Negative F1 score (for minimization)
    """
    model.eval()
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
    return -f1

def evaluate_model_ce_single_batch(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    train: bool = False
) -> float:
    """Evaluate model using cross entropy loss on a single batch.
    
    Args:
        model: Neural network model
        batch: Tuple of (inputs, labels)
        device: Device to run evaluation on
        train: Unused parameter for API compatibility
        
    Returns:
        Cross entropy loss for the batch
    """
    model.eval()
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels).item()
        
    if np.isnan(loss) or np.isinf(loss):
        loss = 9.9e+21  # Handle numerical instability
        
    return loss

# --------------------- Distribution-based Strategy Functions ---------------------

def compute_l2_norm(x: np.ndarray) -> float:
    """Compute L2-norm of x_i.
    
    Args:
        x: Input array of shape (popsize, num_dims)
        
    Returns:
        Mean of squared values
    """
    return np.nanmean(x * x)

def build_model(
    model: nn.Module,
    W: int,
    total_weights: int,
    solution: np.ndarray,
    codebook: Dict[int, np.ndarray],
    state: Dict,
    weight_offsets: np.ndarray,
    device: str = 'cuda'
) -> nn.Module:
    """Build model using solution parameters from distribution-based strategy.
    
    Args:
        model: Base neural network model
        W: Number of components
        total_weights: Total number of parameters
        solution: Solution vector from distribution-based strategy
        codebook: Dictionary mapping components to parameter indices
        state: Distribution-based strategy state
        weight_offsets: Random offsets for parameters
        device: Device to place model on
        
    Returns:
        Updated model with new parameters
    """
    solution = np.array(solution)
    means = solution[:W]
    log_sigmas = solution[W:]
    sigmas = np.exp(log_sigmas)

    params = torch.zeros(total_weights, device=device)
    for k in range(W):
        indices = codebook[k]
        size = len(indices)
        if size > 0:
            mean_tensor = torch.tensor(means[k], device=device)
            sigma_tensor = torch.tensor(sigmas[k], device=device)
            params[indices] = torch.normal(
                mean=mean_tensor,
                std=sigma_tensor,
                size=(size,),
                device=device
            )

    torch.nn.utils.vector_to_parameters(params, model.parameters())
    return model

def train_on_gd(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    step: int = 0,
    warmup_scheduler: WarmUpLR = None,
    args = None,
    device: str = 'cuda'
) -> Tuple[int, float]:
    """Train model using gradient descent.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        criterion: Loss function
        step: Current training step
        warmup_scheduler: Optional warmup learning rate scheduler
        args: Training arguments
        device: Device to run training on
        
    Returns:
        Tuple of (total function evaluations, average loss)
    """
    model.train()
    running_loss = 0.0
    total_fe = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_fe += 1

        if step <= args.warm and warmup_scheduler is not None:
            warmup_scheduler.step()
    
    return total_fe, running_loss / total_fe

# --------------------- Clustering Functions ---------------------

def ubp_cluster(
    W: int,
    params: np.ndarray
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Uniform bin partitioning clustering.
    
    Args:
        W: Number of bins/clusters
        params: Parameters to cluster
        
    Returns:
        Tuple of (codebook, centers, log_sigmas, bin_indices)
    """
    min_val = params.min()
    max_val = params.max()
    bins = np.linspace(min_val, max_val, W)
    bin_indices = np.digitize(params, bins) - 1
    
    centers = []
    log_sigmas = []
    counter = 0
    codebook = {}
    
    for i in range(W):
        mask = np.where(bin_indices == i)[0]
        if len(mask) == 0:
            continue
            
        centers.append(params[mask].mean())
        log_sigmas.append(np.log(params[mask].std() + 1e-8))
        bin_indices[mask] = counter
        codebook[counter] = mask
        counter += 1
        
    return codebook, np.array(centers), np.array(log_sigmas), bin_indices

def random_codebook_initialization(W_init: int, total_weights: int) -> Dict[int, np.ndarray]:
    """Initialize random codebook for clustering.
    
    Args:
        W_init: Number of initial clusters
        total_weights: Total number of parameters
        
    Returns:
        Codebook mapping cluster indices to parameter indices
    """
    weight_indices = np.arange(total_weights)
    np.random.shuffle(weight_indices)
    
    codebook = {}
    d = np.random.dirichlet(np.ones(W_init))
    start_idx = 0
    
    for key in range(W_init):
        size = np.ceil(d[key] * total_weights).astype(int)
        end_idx = start_idx + size
        indices = weight_indices[start_idx:end_idx]
        
        if len(indices) == 0:
            indices = weight_indices[start_idx:end_idx+1]
            
        codebook[key] = indices
        weight_indices[indices] = np.full(len(indices), key)
        start_idx = end_idx
        
    return codebook

# --------------------- Utility Classes ---------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Logger:
    """Custom logger that writes to both terminal and file."""
    
    def __init__(self, log_file: str):
        """Initialize logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()

# --------------------- Training and Visualization Functions ---------------------

def sgd_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    steps: int = 5,
    countfe: int = 0,
    lr: float = 1e-2,
    device: str = 'cuda'
) -> None:
    """Fine-tune model using SGD.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        steps: Number of optimization steps
        lr: Learning rate
        device: Device to run training on
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= steps:
            break
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        countfe += 1

def mse_softmax_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate MSE loss between target class probability and 1.
    
    Args:
        input: Model output logits [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        
    Returns:
        MSE loss: mean((p_target - 1)^2) where p_target is the probability of the correct class
        
    Note:
        This computes MSE only for the target class probability, not all classes.
        - Perfect prediction (p_target=1.0): loss = 0.0
        - Random prediction (p_target=0.1 for 10 classes): loss = 0.81
        - Wrong prediction (p_target=0.0): loss = 1.0
        Dynamic range: [0.0, 1.0] - much better than full MSE's [0.0, 0.18]
    """
    # Apply softmax to logits
    softmax_output = F.softmax(input, dim=1)
    
    # Gather the probabilities of the target classes
    # softmax_output[i, target[i]] for each i in batch
    batch_size = input.shape[0]
    target_probs = softmax_output[torch.arange(batch_size), target]
    
    # Compute MSE between target probabilities and 1
    mse = F.mse_loss(target_probs, torch.ones_like(target_probs))
    return mse


def f1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate F1 loss for model output.
    
    Args:
        input: Model output logits
        target: Ground truth labels
        
    Returns:
        F1 loss (for minimization)
    """
    _, predicted = torch.max(input.data, dim=1)
    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()
    
    f1 = 1 - f1_score(target, predicted, average='macro')
    return torch.tensor(f1, device=input.device)

Averaging = Literal["macro", "micro", "weighted", "none"]

class CombinedCEF1Loss(nn.Module):
    """Combined loss function that combines CrossEntropy and F1 loss with configurable weights.
    
    Args:
        ce_weight: Weight for CrossEntropy loss (default: 0.5)
        f1_weight: Weight for F1 loss (default: 0.5)
        f1_beta: Beta parameter for F1 loss (default: 1.0)
        f1_average: Averaging method for F1 loss (default: 'macro')
        f1_eps: Small epsilon for numerical stability (default: 1e-8)
        normalize: Normalization method for losses ('none', 'log', 'minmax', 'zscore') (default: 'log')
        num_classes: Number of classes for log normalization (default: 10)
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        f1_weight: float = 0.5,
        f1_beta: float = 1.0,
        f1_temperature: float = 1.0,
        f1_learnable_temperature: bool = False,
        f1_average: Averaging = "macro",
        f1_eps: float = 1e-8,
        label_smoothing: float = 0.0,
        normalize: str = "log",
        num_classes: int = 10,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.f1_weight = f1_weight
        self.normalize = normalize
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.f1_loss = SoftF1Loss(
            beta=f1_beta,
            temperature=f1_temperature,
            average=f1_average,
            eps=f1_eps,
            from_logits=True,
            learnable_temperature=f1_learnable_temperature,
            label_smoothing=label_smoothing
        )
        
        # For minmax normalization, we'll track running statistics
        if normalize == "minmax":
            self.register_buffer('ce_min', torch.tensor(float('inf')))
            self.register_buffer('ce_max', torch.tensor(float('-inf')))
            self.register_buffer('f1_min', torch.tensor(float('inf')))
            self.register_buffer('f1_max', torch.tensor(float('-inf')))
            self.register_buffer('update_count', torch.tensor(0))
    
    def _normalize_loss(self, ce_loss: torch.Tensor, f1_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize losses to make them comparable.
        
        Args:
            ce_loss: CrossEntropy loss value
            f1_loss: F1 loss value
            
        Returns:
            Tuple of (normalized_ce_loss, normalized_f1_loss)
        """
        if self.normalize == "none":
            return ce_loss, f1_loss
        
        elif self.normalize == "log":
            # Normalize CE by log(num_classes) to roughly [0, 1] range
            ce_norm = ce_loss / torch.log(torch.tensor(self.num_classes, dtype=ce_loss.dtype, device=ce_loss.device))
            # F1 loss is already in [0, 1] range
            f1_norm = f1_loss
            return ce_norm, f1_norm
        
        elif self.normalize == "minmax":
            # Update running statistics
            with torch.no_grad():
                self.ce_min = torch.min(self.ce_min, ce_loss)
                self.ce_max = torch.max(self.ce_max, ce_loss)
                self.f1_min = torch.min(self.f1_min, f1_loss)
                self.f1_max = torch.max(self.f1_max, f1_loss)
                self.update_count += 1
            
            # Normalize using current min/max
            ce_range = self.ce_max - self.ce_min
            f1_range = self.f1_max - self.f1_min
            
            # Avoid division by zero
            ce_norm = (ce_loss - self.ce_min) / (ce_range + 1e-8)
            f1_norm = (f1_loss - self.f1_min) / (f1_range + 1e-8)
            
            return ce_norm, f1_norm
        
        elif self.normalize == "zscore":
            # For z-score normalization, we'd need running statistics
            # This is a simplified version - in practice, you'd want to track running mean/std
            ce_norm = ce_loss / (torch.log(torch.tensor(self.num_classes, dtype=ce_loss.dtype, device=ce_loss.device)) + 1e-8)
            f1_norm = f1_loss
            return ce_norm, f1_norm
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize}")
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            input: Model predictions (logits)
            target: Ground truth labels
            
        Returns:
            Combined loss value
        """
        ce_loss = self.ce_loss(input, target)
        f1_loss = self.f1_loss(input, target)
        
        # Normalize losses if requested
        ce_norm, f1_norm = self._normalize_loss(ce_loss, f1_loss)
        
        return self.ce_weight * ce_norm + self.f1_weight * f1_norm


class SoftF1Loss(nn.Module):
    """
    Soft F1 loss using softmax probabilities.
    - Supports multi-class classification.
    - Computes macro-averaged (default) or micro-averaged soft-F1.
    - Can be used as a loss (1 - F1) or as a direct F1 score.

    Args:
        average: 'macro' or 'micro'
        eps: numerical stability
        loss: if True -> returns loss = 1 - softF1, else returns softF1 score.
        temperature: optional softmax temperature (T < 1 sharpens, T > 1 smooths)
    """
    def __init__(self, average='macro', eps=1e-8, loss=True, temperature=1.0):
        super().__init__()
        self.average = average
        self.eps = eps
        self.loss = loss
        self.temperature = temperature

    def forward(self, logits, targets):
        """
        logits: (N, C) raw output from model
        targets: (N,) integer class labels 0..C-1
        """
        # Apply temperature-scaled softmax
        probs = F.softmax(logits / self.temperature, dim=1)  # (N, C)

        N, C = probs.shape
        one_hot = F.one_hot(targets, num_classes=C).float()  # (N, C)

        # Soft counts
        tp = (probs * one_hot).sum(dim=0)        # (C,)
        pred_pos = probs.sum(dim=0)              # (C,)
        actual_pos = one_hot.sum(dim=0)          # (C,)

        precision = tp / (pred_pos + self.eps)
        recall = tp / (actual_pos + self.eps)

        f1_per_class = 2 * precision * recall / (precision + recall + self.eps)

        if self.average == 'macro':
            f1 = f1_per_class.mean()
        elif self.average == 'micro':
            # compute micro precision/recall
            tp_micro = tp.sum()
            pred_pos_micro = pred_pos.sum()
            actual_pos_micro = actual_pos.sum()
            prec = tp_micro / (pred_pos_micro + self.eps)
            rec = tp_micro / (actual_pos_micro + self.eps)
            f1 = 2 * prec * rec / (prec + rec + self.eps)
        else:
            raise ValueError("average must be 'macro' or 'micro'")

        if self.loss:
            return 1.0 - f1  # minimize
        else:
            return f1       # maximize or monitor

class SoftBetaF1Loss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        from_logits: bool = True,
        multilabel: bool = False,
        average: Averaging = "macro",
        eps: float = 1e-8,
        class_weights: Optional[torch.Tensor] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        min_temperature: float = 1e-3,
        max_temperature: float = 1e3,
        label_smoothing: float = 0.0,
    ):
        """
        temperature: initial temperature (applied as logits / temperature)
        learnable_temperature: if True, temperature is a learnable scalar parameter
        min_temperature/max_temperature: clamps learned temperature to this range
        label_smoothing: label smoothing factor for better generalization
        """
        super().__init__()
        assert beta > 0
        assert average in ("macro", "micro", "weighted", "none")
        assert reduction in ("mean", "sum", "none")
        self.beta = float(beta)
        self.from_logits = bool(from_logits)
        self.multilabel = bool(multilabel)
        self.average = average
        self.eps = float(eps)
        self.reduction = reduction
        self.register_buffer("class_weights_buffer", None if class_weights is None else class_weights.clone().float())

        # temperature handling
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        if learnable_temperature:
            # store log-temperature to ensure positivity
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
            self._learnable_temperature = True
        else:
            self.register_buffer("fixed_temperature", torch.tensor(float(temperature)))
            self._learnable_temperature = False
            
        # Label smoothing parameter
        self.label_smoothing = label_smoothing

    def _get_temperature(self):
        if self._learnable_temperature:
            t = torch.exp(self.log_temperature)
            # clamp for stability (in-place clamp risks gradient issues; use .clamp)
            return t.clamp(min=self.min_temperature, max=self.max_temperature)
        else:
            return self.fixed_temperature

    def _apply_label_smoothing(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to targets."""
        if self.label_smoothing <= 0.0:
            return y
            
        n_classes = y.shape[1]
        smooth_y = y * (1.0 - self.label_smoothing) + self.label_smoothing / n_classes
        return smooth_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # apply temperature only if from_logits=True and temperature != 1
        T = self._get_temperature()
        if self.from_logits:
            if T.item() != 1.0:
                input = input / T

            if self.multilabel:
                probs = torch.sigmoid(input)
            else:
                probs = F.softmax(input, dim=1)
        else:
            probs = input.float()

        # the rest is identical soft-F1 computation:
        if probs.dim() == 1:
            probs = probs.unsqueeze(1)
        n_classes = probs.shape[1]

        # prepare targets
        if not self.multilabel:
            if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
                if target.dim() == 2:
                    target = target.squeeze(1)
                y = F.one_hot(target.long(), num_classes=n_classes).float()
            else:
                y = target.float()
        else:
            y = target.float()

        y = y.to(probs.dtype)
        if y.shape != probs.shape:
            raise ValueError(f"target shape {y.shape} != input/probs shape {probs.shape}")

        # Apply label smoothing
        y = self._apply_label_smoothing(y)

        TP = (probs * y).sum(dim=0)
        FP = (probs * (1 - y)).sum(dim=0)
        FN = ((1 - probs) * y).sum(dim=0)

        b2 = self.beta * self.beta
        numer = (1.0 + b2) * TP
        denom = (1.0 + b2) * TP + b2 * FN + FP
        per_class_f = numer / (denom + self.eps)

        if self.average == "micro":
            TP_sum = TP.sum()
            FP_sum = FP.sum()
            FN_sum = FN.sum()
            numer_m = (1.0 + b2) * TP_sum
            denom_m = (1.0 + b2) * TP_sum + b2 * FN_sum + FP_sum
            f_val = numer_m / (denom_m + self.eps)
            final_f = f_val
        else:
            f = per_class_f
            # class-weighting (provided or computed)
            class_weights = self.class_weights_buffer
            if class_weights is not None:
                w = class_weights.to(probs.device).float()
                if w.numel() != n_classes:
                    raise ValueError("class_weights length mismatch")
                final_f = (f * w).sum() / (w.sum() + self.eps)
            elif self.average == "weighted":
                support = target.sum(dim=0)
                total_support = support.sum()
                if total_support.item() == 0:
                    final_f = f.mean()
                else:
                    weights = support / (total_support + self.eps)
                    final_f = (f * weights).sum()
            elif self.average == "none":
                final_f = f
            else:  # macro
                final_f = f.mean()

        loss = 1.0 - final_f

        if self.average == "none":
            if self.reduction == "none":
                return loss
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss.mean()
        else:
            if self.reduction == "none":
                return loss
            elif self.reduction == "sum":
                return loss * 1.0
            else:
                return loss


def accuracy(input: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = input.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# --------------------- Parameter Management Functions ---------------------

def get_param_shapes(model: nn.Module) -> List[torch.Size]:
    """Get shapes of all trainable parameters in model.
    
    Args:
        model: Neural network model
        
    Returns:
        List of parameter shapes
    """
    return [param.shape for param in model.parameters() if param.requires_grad]

def params_to_vector(params: List[torch.Tensor], to_numpy: bool = False) -> torch.Tensor:
    """Flatten parameters into single vector.
    
    Args:
        params: List of parameter tensors
        
    Returns:
        Flattened parameter vector
    """
    params_vector = torch.nn.utils.parameters_to_vector(params).detach()
    if to_numpy:
        return params_vector.cpu().numpy()
    return params_vector

def assign_flat_params(model: nn.Module, params: torch.Tensor) -> None:
    """Assign flattened parameters back to model.
    
    Args:
        model: Neural network model
        params: Flattened parameter vector
    """
    parameters = model.parameters()
    # Ensure vec of type Tensor
    if not isinstance(params, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(flat_params)}")

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        if not param.requires_grad:
            continue
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = params[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def freeze_bn(model: nn.Module) -> None:
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None

    # for name, param in model.named_parameters():
    #         if "bn" in name:
    #             param.requires_grad = False

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

def unfreeze_bn(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "bn" in name:
            param.requires_grad = True



def fitness(z, model, base_params, decoder, batch, loss_fn, device, alpha):
    """Compute fitness of a solution.
    
    Args:
        z: Latent vector to evaluate.
        model: Model to evaluate.
        base_params: Base parameters to use.
        decoder: Decoder to use.
        batch: Batch of data.
        loss_fn: Loss function.
        device: Device to use.
        
    Returns:
        float: Fitness value (loss).
    """
    model.eval()
    decoder.apply(model=model, z=z, base_params=base_params, alpha=alpha)
    total_loss = 0.0
    
    with torch.no_grad():
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(x)
        total_loss += loss_fn(output, y).item()
        
    return total_loss


def plot_trajectory(param_trajectory, sample_trajectory=None, save_path=None):
    # Convert trajectory to numpy array
    param_trajectory = np.array(param_trajectory)

    # Fit PCA and transform trajectory
    pca = PCA(n_components=2)
    pca.fit(param_trajectory)
    trajectory_2d = pca.transform(param_trajectory)

    if sample_trajectory is not None:
        sample_trajectory_2d = pca.transform(sample_trajectory)

    # Plot the trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'b-', label='Parameter Trajectory')
    plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                c=range(1, len(trajectory_2d)+1), cmap='viridis', 
                label='Training Progress', zorder=20)
    if sample_trajectory is not None:
        plt.scatter(sample_trajectory_2d[:, 0], sample_trajectory_2d[:, 1],
                    c='red', alpha=0.5, 
                    zorder=10, label='Gaussian Sampling on Parameter Trajectory')
    plt.colorbar(label='Training Progress')
    plt.title('Parameter Trajectory During Training (PCA Projection)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_3d_trajectory(param_trajectory, loss_trajectory, sample_trajectory=None, sample_loss_trajectory=None, save_path=None):
    # Convert trajectory to numpy array
    param_trajectory = np.array(param_trajectory)
    loss_trajectory = np.array(loss_trajectory)
    
    # Ensure positive values for log scale
    loss_trajectory = np.abs(loss_trajectory) + 1e-10  # Add small epsilon to avoid zero
    if sample_loss_trajectory is not None:
        sample_loss_trajectory = np.abs(sample_loss_trajectory) + 1e-10

    # Fit PCA and transform trajectory
    pca = PCA(n_components=2)
    pca.fit(param_trajectory)
    trajectory_2d = pca.transform(param_trajectory)

    if sample_trajectory is not None:
        sample_trajectory_2d = pca.transform(sample_trajectory)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    scatter = ax.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], loss_trajectory,
                        c=range(len(trajectory_2d)), cmap='viridis', zorder=20, s=100,
                        label='Training Progress')
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], loss_trajectory, 'b-', 
            alpha=0.3, zorder=15, label='Parameter Trajectory')

    if sample_trajectory is not None:
        ax.scatter(sample_trajectory_2d[:, 0], sample_trajectory_2d[:, 1], sample_loss_trajectory, 
                   c='red', alpha=0.5, zorder=10, s=10,
                   label='Gaussian Sampling on Parameter Trajectory')

    # Add colorbar
    plt.colorbar(scatter, label='Training Progress')

    # Set labels and title
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Loss Value (log scale)')
    ax.set_title('Parameter Trajectory During Training (3D PCA Projection)')

    # Set z-axis to log scale
    ax.set_zscale('log')

    # Add legend
    ax.legend()

    # Set the view angle
    ax.view_init(elev=30, azim=45)

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_param_distribution(x, save_path):
    
    # Plot histogram of x
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(x, bins=100, density=True, alpha=0.6, color='g', label='Parameters')

    # Plot the theoretical normal distribution curve
    mu, std = np.mean(x), np.std(x)
    pdf = norm.pdf(bins, mu, std)
    plt.plot(bins, pdf, 'k--', linewidth=2, label='Normal PDF')

    plt.title('Parameter distribution vs. Normal distribution')
    plt.xlabel(f'$\\theta$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}")
    plt.close()


class ConfidenceMarginLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        softmax_output = F.softmax(input, dim=1)
        
        # Get the probability of the target class (correct indexing)
        batch_size = target.size(0)
        p_target = softmax_output[torch.arange(batch_size, device=input.device), target]
        
        # Mask out the target class to find the maximum probability among incorrect classes
        mask = torch.ones_like(softmax_output, dtype=torch.bool)
        mask[torch.arange(batch_size, device=input.device), target] = False
        p_max_incorrect = softmax_output[mask].view(batch_size, -1).max(dim=1)[0]
        
        # Margin-based cross-entropy:
        # We want to maximize p_c while minimizing p_max_incorrect
        # 
        # Loss components:
        # 1. -log(p_c): standard cross-entropy (encourages high p_c)
        # 2. -log(1 - p_max_incorrect): penalizes high incorrect probabilities
        # 
        # Combined: -log(p_c) - log(1 - p_max_incorrect)
        #         = -log(p_c * (1 - p_max_incorrect))
        # 
        # Properties:
        # - Always positive (log of probability)
        # - Unbounded above (good for ranking)
        # - When p_c→1 and p_max_incorrect→0: loss→0 (perfect)
        # - When p_c is low OR p_max_incorrect is high: loss is large
        eps = 1e-7
        loss = -torch.log(p_target + eps) - torch.log(1 - p_max_incorrect + eps)
        
        return loss.mean()


class TotalIncorrectPenaltyLoss(nn.Module):
    """
    Loss function that maximizes target class probability while minimizing 
    the sum of all incorrect class probabilities.
    
    Loss = -log(p_c) + sum_{i≠c} p_i
         = -log(p_c) + (1 - p_c)
    
    This provides:
    - Logarithmic encouragement for high p_c (unbounded as p_c → 0)
    - Linear penalty for incorrect probabilities (bounded between 0 and 1)
    - Good dynamic range for CMA-ES ranking
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        softmax_output = F.softmax(input, dim=1)
        
        # Get the probability of the target class
        batch_size = target.size(0)
        p_target = softmax_output[torch.arange(batch_size, device=input.device), target]
        
        # Sum of all incorrect class probabilities
        # Since sum of all probabilities = 1, sum of incorrect = 1 - p_target
        sum_incorrect = 1 - p_target
        
        # Loss: cross-entropy term + linear penalty on incorrect probabilities
        # -log(p_c) encourages high target probability (unbounded)
        # sum_{i≠c} p_i = (1 - p_c) penalizes incorrect probabilities (bounded)
        eps = 1e-7
        loss = -torch.log(p_target + eps) + sum_incorrect
        
        return loss.mean()


def create_criterion(args, num_classes):
    """
    Create a loss criterion and optionally a new train loader with weighted sampling.
    
    Args:
        args: Arguments object containing criterion configuration
        num_classes: Number of classes in the dataset
        
    Returns:
        tuple: (criterion, train_loader or None)
               Returns None for train_loader if no weighted sampling is needed
    """
    
    criterion_type = args.criterion.lower()
    
    if criterion_type == 'ce':
        if args.label_smoothing is None:
            args.label_smoothing = 0.0
        criterion = lambda input, target: F.cross_entropy(input, target, label_smoothing=args.label_smoothing).item()
        
    elif criterion_type == 'f1':
        criterion = f1_loss
        
    elif criterion_type == 'mse':
        criterion = mse_softmax_loss
        
    elif criterion_type == 'soft_f1':
        if args.f1_temperature is None:
            args.f1_temperature = 1.0
        criterion = SoftF1Loss(
            average="micro",
            loss=True,
            temperature=args.f1_temperature,
        )
    elif criterion_type == 'cm':
        criterion = ConfidenceMarginLoss(
            num_classes=num_classes
        )
    
    elif criterion_type == 'tip':
        criterion = TotalIncorrectPenaltyLoss(
            num_classes=num_classes
        )
        
    elif criterion_type == 'ce_sf1':
        if args.ce_weight is None:
            args.ce_weight = 0.5
        if args.f1_weight is None:
            args.f1_weight = 0.5
        if args.f1_beta is None:
            args.f1_beta = 1.0
        if args.f1_temperature is None:
            args.f1_temperature = 1.0
        if args.f1_learnable_temperature is None:
            args.f1_learnable_temperature = False
        if args.ce_normalize is None:
            args.ce_normalize = 'log'
        if args.label_smoothing is None:
            args.label_smoothing = 0.0
            
        criterion = CombinedCEF1Loss(
            ce_weight=args.ce_weight,
            f1_weight=args.f1_weight,
            f1_beta=args.f1_beta,
            f1_temperature=args.f1_temperature,
            f1_learnable_temperature=args.f1_learnable_temperature,
            label_smoothing=args.label_smoothing,
            normalize=args.ce_normalize,
            num_classes=num_classes
        )
        
    else:
        raise ValueError(f"Invalid criterion: {criterion_type}")
    
    return criterion



def evaluate_model_on_batch(model, criterion, batch, device):
    model.eval()
    model = model.to(device)
    try:
        with torch.no_grad():
            inputs, targets = batch[0].to(device), batch[1].to(device)
            output = model(inputs)
            loss = criterion(output, targets)
        return loss.item()
    except RuntimeError as e:
        if "CUDA" in str(e):
            # Clear CUDA cache and retry
            torch.cuda.empty_cache()
            print(f"CUDA error encountered, cleared cache: {e}")
            return 1e6  # Return high loss value as fallback
        else:
            raise e

def load_solution_to_model(z, ws, device):
    try:
        theta = ws(z)
        if torch.cuda.is_available() and device == 'cuda':
            theta = theta.to(device)
        ws.load_to_model(theta)
    except RuntimeError as e:
        if "CUDA" in str(e):
            torch.cuda.empty_cache()
            print(f"CUDA error in load_solution_to_model, cleared cache: {e}")
            # Retry once with cache cleared
            theta = ws(z)
            if torch.cuda.is_available() and device == 'cuda':
                theta = theta.to(device)
            ws.load_to_model(theta)
        else:
            raise e


def evaluate_solution_on_batch(z, ws, criterion, batch, weight_decay=0, device='cuda'):
    load_solution_to_model(z, ws, device)
    fitness = evaluate_model_on_batch(model=ws.model, criterion=criterion, batch=batch, device=device)
    theta = params_to_vector(ws.model.parameters(), to_numpy=True)
    theta_norm = np.linalg.norm(theta)
    fitness = fitness + weight_decay * theta_norm
    return fitness


def evaluate_population_on_batch(population, adapter, criterion, batch, train_loader=None, weight_decay=0, device='cuda'):
    fitnesses = np.zeros(len(population))
    for i, z in enumerate(population):
        if train_loader is not None:
            batch = next(iter(train_loader))
        try:
            load_solution_to_model(z, adapter, device)
            fitnesses[i] = evaluate_model_on_batch(model=ws.model, criterion=criterion, batch=batch, device=device)
            theta = params_to_vector(adapter.model.parameters(), to_numpy=True)
            theta_norm = np.linalg.norm(theta)
            fitnesses[i] = fitnesses[i] + weight_decay * theta_norm
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Clear CUDA cache and assign high fitness value
                torch.cuda.empty_cache()
                print(f"CUDA error in population evaluation {i}, cleared cache: {e}")
                fitnesses[i] = 1e6  # High fitness value (bad for minimization)
            else:
                raise e
        
        # Clear cache periodically to prevent memory buildup
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    return fitnesses