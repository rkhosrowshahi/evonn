import argparse
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import get_model
from utils import WarmUpLR, load_data, save_model, set_seed, init_wandb, log_training_metrics, finish_wandb_run, evaluate_model

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated


def train_with_sgd(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize wandb with consistent configuration
    if not args.disable_wandb:
        config = {
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "device": device,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "warm": args.warm,
            "seed": args.seed,
            "trainer_type": "sgd"
        }
        
        run_name = args.wandb_name if args.wandb_name else f"{args.model}_{args.dataset}_sgd_ep{args.epochs}"
        init_wandb(
            project_name=args.wandb_project,
            run_name=run_name,
            config=config,
            tags=[args.model, args.dataset, "sgd"]
        )
    set_seed(args.seed)
    train_dataset, val_dataset, test_dataset, num_classes = load_data(args.dataset, validation_split=0.01)
    
    # Create data loaders
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    # Initialize model
    model = get_model(model_name=args.model, num_classes=num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Update wandb config with model info
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if wandb.run is not None:
        wandb.config.update({'num_weights': num_weights}, allow_val_change=True)
    initial_loss, initial_top1 = evaluate_model(model=model, data_loader=test_loader, criterion=criterion, device=device, train=False)
    print(f"Initial training loss: {initial_loss:.4f}, top1: {initial_top1:.2f}%")

    # Training Loop
    pbar = tqdm(range(1, args.epochs + 1), desc="Training")
    for epoch in pbar:
        # Train for one epoch
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        # Step learning rate scheduler after warmup
        scheduler.step()
            
        # Evaluate at end of epoch
        train_loss, train_top1 = evaluate_model(model=model, data_loader=train_loader, criterion=criterion, device=device, train=False)
        test_loss, test_top1 = evaluate_model(model=model, data_loader=test_loader, criterion=criterion, device=device, train=False)
        
        # Log training metrics to wandb
        if wandb.run is not None:
            log_training_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                test_loss=test_loss,
                test_top1=test_top1,
                additional_metrics={
                    'lr': optimizer.param_groups[0]['lr'],
                }
            )
        
        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}", 
            "train_top1": f"{train_top1:.2f}%",
            "test_loss": f"{test_loss:.4f}", 
            "test_top1": f"{test_top1:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    # Final Evaluation
    final_train_loss, final_train_accuracy = evaluate_model(model=model, criterion=criterion, data_loader=train_loader, device=device, train=False)
    final_test_loss, final_test_accuracy = evaluate_model(model=model, criterion=criterion, data_loader=test_loader, device=device, train=False)
    
    print(f"Final Train Accuracy: {final_train_accuracy:.2f}%")
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
    
    if wandb.run is not None:
        wandb.log({
            "final_train_accuracy": final_train_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss
        })
        save_model(model, f"{args.dataset}_{args.model}_{args.lr}_{args.batch_size}_{args.epochs}_{args.warm}", wandb)

    # Finish wandb run
    finish_wandb_run()

# Run Training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warm", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='evo-training', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto-generated if None)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    train_with_sgd(args)