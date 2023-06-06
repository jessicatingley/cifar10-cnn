import torch
import argparse
import torchvision.transforms as transforms
from engine import *


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_layers', type=int,)
    parser.add_argument('num_neurons', type=int,)
    parser.add_argument('activation_func', type=str,)
    parser.add_argument('learning_rate', type=float,)
    parser.add_argument('dropout', type=float,)
    parser.add_argument('batch_size', type=int,)
    parser.add_argument('optimizer', type=str,)
    parser.add_argument('epochs', type=int,)
    parser.add_argument('depth', type=int,)
    parser.add_argument('kernel_size', type=int,)
    parser.add_argument('block_kernel_size', type=int)
    parser.add_argument('num_blocks', type=int)
    return parser

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = make_parser()
    args = parser.parse_args()

    net = create_net(depth=args.depth, kernel_size=args.kernel_size, block_kernel_size=args.block_kernel_size, 
                     num_blocks=args.num_blocks, num_neurons=args.num_neurons, activation_func=args.activation_func, 
                     dropout=args.dropout)
    
    train_loader, val_loader, test_loader = load_data(batch_size=args.batch_size, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    train_net(net=net, trainloader=train_loader, device=device, learn_rate=args.learning_rate, 
              given_optimizer=args.optimizer, epochs=args.epochs, val_loader=val_loader, criterion=criterion, patience=3)

    train_accuracy, train_loss = evaluate_net(net, train_loader, device, criterion)
    print("Train Accuracy: {:.2f}%".format(train_accuracy))
    print("Train Loss: {:.4f}".format(train_loss))

    test_accuracy, test_loss = evaluate_net(net, test_loader, device, criterion)
    print("Test Accuracy: {:.2f}%".format(test_accuracy))
    print("Test Loss: {:.4f}".format(test_loss))

    val_accuracy, val_loss = evaluate_net(net, val_loader, device, criterion)
    print("Validation Accuracy: {:.2f}%".format(val_accuracy))
    print("Validation Loss: {:.4f}".format(val_loss))


if __name__ == "__main__":
    main()
