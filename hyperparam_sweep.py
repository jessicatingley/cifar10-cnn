import wandb
import torch
import torch.nn as nn
from engine import *


sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters":
    {
        "num_layers": {"value": 3},
        "num_neurons": {"values": [512, 2048]},
        "activation_func": {"values": ['relu', 'sigmoid']},
        "learning_rate": {"values": [0.0001, 0.001, 0.01, 0.1]},
        "dropout": {"values": [0, 0.1]},
        "batch_size": {"value": 512},
        "optimizer": {"value": "adam"},
        "epochs": {"value": 50},
        "depth": {"values": [16, 64]},
        "kernel_size": {"values": [5, 7]},
        "block_kernel_size": {"values": [3, 5]},
        "num_blocks": {"values": [3, 4, 5]}
    }
}


sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="robotics-nn"
)


def main():
    run = wandb.init()

    num_layers = wandb.config.num_layers
    num_neurons = wandb.config.num_neurons
    activation_func = wandb.config.activation_func
    learning_rate = wandb.config.learning_rate
    dropout = wandb.config.dropout
    batch_size = wandb.config.batch_size
    optimizer = wandb.config.optimizer
    epochs = wandb.config.epochs
    depth = wandb.config.depth
    kernel_size = wandb.config.kernel_size
    block_kernel_size = wandb.config.block_kernel_size
    num_blocks = wandb.config.num_blocks

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, num_workers=2)
    model = create_net(depth, kernel_size, block_kernel_size, num_blocks, num_neurons, activation_func, dropout)
    train_net(model, trainloader=train_loader, device=device, learn_rate=learning_rate, given_optimizer=optimizer, 
              epochs=epochs, val_loader=val_loader, criterion=nn.CrossEntropyLoss(), patience=3)
    train_accs, train_losses = evaluate_net(model, train_loader, device, nn.CrossEntropyLoss())
    val_acc, val_loss = evaluate_net(model, val_loader, device, nn.CrossEntropyLoss())
    
    for i, (train_loss, train_acc) in enumerate(zip(train_losses, train_accs)):
        wandb.log({
            "epoch" : i,
            "train_acc" : train_acc,
            "train_loss" : train_loss,
        })

    wandb.log({
        "val_acc": val_acc, 
        "val_loss": val_loss,
        })
    
if __name__ == "__main__":
    wandb.agent(sweep_id=sweep_id, function=main)
