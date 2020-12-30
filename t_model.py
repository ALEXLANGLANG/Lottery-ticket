from prune_layer import *
from archs.mnist import  LeNet5
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils

# Function for Testing
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

# testdataset = datasets.MNIST('/work/data/Xian', train=False, transform=transform)

# test_loader = torch.utils.data.DataLoader(testdataset, batch_size=128, shuffle=False, num_workers=0,drop_last=True)
# criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

for i  in range(5):
    PATH = 'saves/lenet5/mnist/'
    PATH += str(i) + '_model_lt.pt'
    model = LeNet5.LeNet5()
    model= torch.load(PATH)
    utils.print_nonzeros(model)



