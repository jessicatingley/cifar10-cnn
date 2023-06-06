import torch
import torchinfo
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
import numpy as np
import random
from sklearn.model_selection import train_test_split


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class LayerBlock(nn.Module):
    def __init__(self, depth: int, kernel_size: int, activation_func: str, dropout: float):
        super(LayerBlock, self).__init__()
        self.depth = depth
        map_activ_func = {"relu" : nn.ReLU(), "sigmoid" : nn.Sigmoid(), "elu" : nn.ELU()}

        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=depth*2, kernel_size=kernel_size, padding='same')
        self.activation = map_activ_func[activation_func]
        self.conv1_norm = nn.BatchNorm2d(num_features=depth*2) 
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(depth * 2, depth, kernel_size=kernel_size, padding='same')
        self.conv2_norm = nn.BatchNorm2d(num_features=depth)
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, inputs):
        x = self.activation(self.conv1(inputs))
        x = self.conv1_norm(x)
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.conv2_norm(x)
        x = self.dropout(x)
        x = x + inputs
        x = self.maxpool(x)
        return x


class Net(nn.Module):
    def __init__(self, depth: int, kernel_size: int, block_kernel_size: int, num_blocks: int, num_neurons: int, activation_func: str, dropout: float):
        super(Net, self).__init__()
        self.depth = depth
        self.num_blocks = num_blocks
        map_activ_func = {"relu" : nn.ReLU(), "sigmoid" : nn.Sigmoid(), "elu" : nn.ELU()}
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=depth, kernel_size=kernel_size, padding="same")
        self.activation = map_activ_func[activation_func]
        self.norm = nn.BatchNorm2d(depth)
        self.blocks = nn.ParameterList()
        for _ in range(num_blocks):
            self.blocks.append(LayerBlock(depth=depth, kernel_size=block_kernel_size, activation_func=activation_func, dropout=dropout,))
        self.fc1 = nn.Linear(in_features=depth * (32 // (2 ** self.num_blocks)) ** 2, out_features=num_neurons)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=num_neurons, out_features=10)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        x = x.view(-1, self.depth * (32 // (2 ** self.num_blocks)) ** 2)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    full_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
    
    train_indices, val_indices = train_test_split(list(range(len(full_dataset))), 
                                                  test_size=0.2,
                                                  random_state=42)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, 
                                           download=True, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def create_net(depth: int, kernel_size: int, block_kernel_size: int, num_blocks:int, num_neurons: int, activation_func: str, dropout: float,):
    net = Net(depth=depth, kernel_size=kernel_size, block_kernel_size=block_kernel_size, 
              num_blocks=num_blocks, num_neurons=num_neurons, activation_func=activation_func, dropout=dropout)
    #torchinfo.summary(net, (3, 32, 32), batch_dim=0, col_names= ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1)
    return net


def train_net(net, trainloader, device, learn_rate: float, given_optimizer: str, epochs: int, val_loader, criterion, patience):
    criterion = nn.CrossEntropyLoss()
    map_optimizer = {"sgd" : optim.SGD(params=net.parameters(), lr=learn_rate, momentum=0.9), "adam" : optim.Adam(params=net.parameters(), lr=learn_rate)}
    optimizer = map_optimizer[given_optimizer]

    net = net.to(device)

    best_loss = float('inf')
    epochs_since_best_loss = 0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        _, val_loss = evaluate_net(net=net, loader=val_loader, device=device, criterion=criterion)
        print("Best Loss: {}".format(best_loss))
        print("Current Val Loss: {}".format(val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_best_loss = 0
            print("Epochs since best loss: 0")
        else:
            epochs_since_best_loss +=1
            print("Epochs since best loss: {}".format(epochs_since_best_loss))
        
        if epochs_since_best_loss >= patience:
            print("Validation loss hasn't improved in {} epochs. Stopping early.".format(patience))
        


def evaluate_net(net, loader, device, criterion):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = total_loss / len(loader)
    return accuracy, average_loss
