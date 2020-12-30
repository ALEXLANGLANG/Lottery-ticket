# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns

import torch.nn.init as init
import pickle
from prune_layer import *
writer = SummaryWriter()

import time
# Custom Libraries
import utils

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

                
                
def prune_percentage_nonzero(q = 10):
    global model
    for n,m in model.named_modules():
        if isinstance(m, PrunedConv):
            m.prune_by_percentage(q = q)
        if isinstance(m, PruneLinear):
            m.prune_by_percentage(q = q)
            
def mask_weights(mask_data = True): 
    global model 
    if mask_data:
        for n, m in model.named_modules():
            if isinstance(m, PrunedConv):
                m.conv.weight.data.mul_(m.mask)
            if isinstance(m, PruneLinear):
                m.linear.weight.data.mul_(m.mask)
    else:
        for n, m in model.named_modules():
            if isinstance(m, PrunedConv):
                m.conv.weight.grad.mul_(m.mask)
            if isinstance(m, PruneLinear):
                m.linear.weight.grad.mul_(m.mask)
            
def initialize_weights(initial_state_dict):
    global model 
    for n,m in model.named_modules():
        if isinstance(m, PrunedConv):
            m.mask[torch.abs(m.mask) < 1.] = 1
            m.conv.weight.data = m.mask*initial_state_dict[n + '.conv.weight']
            m.conv.weight.bias = initial_state_dict[n + '.conv.bias']
        if isinstance(m, PruneLinear):
            m.mask[torch.abs(m.mask) < 1.] = 1
            m.linear.weight.data = m.mask*initial_state_dict[n + '.linear.weight']
            m.linear.weight.bias =initial_state_dict[n + '.linear.bias']
            
def reintilize_weights(weight_init):
    global model 
    model.apply(weight_init)
    mask_weights()


# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        mask_weights()#Mask data into zero
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        mask_weights(False) #Mask gradients of weights to zero
        optimizer.step()
    return train_loss.item()


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

def get_mask_all():
    global model
    mask = []
    for n, m in model.named_modules():
        if isinstance(m, PrunedConv):
            mask += [m.mask.cpu().numpy()]
        if isinstance(m, PruneLinear):
            mask += [m.mask.cpu().numpy()]
    return mask



def run_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ITE=0
    reinit = True if args.prune_type=="reinit" else False

    #Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('/local/data/Xian', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('/local/data/Xian', train=False, transform=transform)
        from archs.mnist import  LeNet5, fc1, vgg, resnet,AlexNet
    # If you want to add extra datasets paste here
    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
        model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    else:
        print("\nWrong Model choice\n")
        exit()

    model.apply(weight_init)
    # Copying and Saving Initial State


    model = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pt")
    initial_state_dict = copy.deepcopy(model.state_dict())

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss


    with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat", 'rb') as f:
        compress_rate = pickle.load(f)
    # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())


    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = len(compress_rate)
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
    args.prune_iterations = ITERATION
    print(compress_rate)
    print(100. - compress_rate)
    for _ite in range(args.start_iter, ITERATION):
        initialize_weights(initial_state_dict)        
        prune_percentage_nonzero(q = 100. - compress_rate[_ite])
        mask_weights()
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}]: un_pruned: {compress_rate[_ite]}%---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/oneshot/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/oneshot/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/oneshot/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/oneshot/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")

        # Dumping mask
        mask = get_mask_all()
        utils.checkdir(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/oneshot/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/oneshot/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/oneshot/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()


    
    
    
class argument:
    def __init__(self, lr=1.2e-3,batch_size = 128,start_iter = 0,end_iter = 100,print_freq = 1,
                 valid_freq = 1,resume = "store_true",prune_type= "lt",gpu = "0",
                 dataset = "mnist" ,arch_type = "fc1",prune_percent  = 10,prune_iterations = 35):
            self.lr = lr
            self.batch_size = batch_size
            self.start_iter = start_iter
            self.end_iter = end_iter
            self.print_freq = print_freq
            self.valid_freq = valid_freq
            self.resume = resume
            self.prune_type = prune_type #reinit
            self.gpu = gpu
            self.dataset = dataset #"mnist | cifar10 | fashionmnist | cifar100"
            self.arch_type = arch_type  # "fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")

args = argument(end_iter = 20,arch_type ="lenet5")
run_main(args)
