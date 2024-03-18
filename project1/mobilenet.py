import random
import argparse
import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from dataloader import VehicleDataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import time
import os

# TODO:
# Decide your own hyper-parameters
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
NUM_EPOCH = 10
NUM_CLASSES = 7
def train_one_batch(
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: Optimizer, 
        images: torch.Tensor, 
        labels: torch.Tensor
        ) -> None:
    images = images.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
def train(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
):
    model.train()
    for images, labels in tqdm(dataloader, desc='Training'):
        images = images.cuda()
        labels = labels.cuda()
        train_one_batch(model, criterion, optimizer, images, labels)

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    result_root: str
) -> float:
    model.eval()
    num_samples = 0
    num_corrects = 0
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for images, labels in tqdm(val_loader, desc='Evaluating'):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicts = torch.max(outputs, dim=1)
        num_samples += labels.size(0) # labels.size(0) is batch size
        num_corrects += (predicts == labels).sum()
        for i in range(labels.size(0)): # batch size
            confusion_matrix[labels[i]][predicts[i]] += 1
    accuracy = num_corrects / num_samples
    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix)
    # Create a heatmap from the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues', 
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    # Save the figure
    plt.savefig(os.path.join(result_root, f'confusion_{accuracy:.4f}.jpg'), dpi=300)
    plt.close()
    return accuracy


def Trainer(
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        result_root: str,
        ckpt_root: str
) -> None: 
    # Send the model from cpu to gpu
    best_acc = 0
    best_model = None

    for epoch in tqdm(range(1, NUM_EPOCH+1)):
        train(model, train_loader, criterion, optimizer)
        acc = evaluate(model, val_loader, result_root)
        print(f'epoch, {epoch}, accuracy: {acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            best_model = model.state_dict()

    print(f'Finished Training, best accuracy: {best_acc:.4f}, learning rate: {LEARNING_RATE}')
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Save the best model
    # ckpt_name = f'/home/aa35037123/Wesley/ai_capstone/project1/ckpt/model_{timestamp}.ckpt'
    model_ckpt_path = os.path.join(ckpt_root, f'model_{timestamp}.pt')
    torch.save(best_model, model_ckpt_path)
    opt_ckpt_path = os.path.join(ckpt_root, f'opt_{timestamp}.pt')
    torch.save(optimizer.state_dict(), opt_ckpt_path)
    print(f'Save model to {model_ckpt_path}')
    print(f'Save optimizer to {opt_ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True) # add help to the parser
    parser.add_argument('--ckpt_model', type=str, default=None,     help="path of model pretrain weight")
    parser.add_argument('--ckpt_opt', type=str, default=None,     help="path of optimizer pretrain weight")
    parser.add_argument('--lr',    type=float,    default=LEARNING_RATE)
    parser.add_argument('--mode',    type=str,    default="train")
    args = parser.parse_args()
    # To ensure the reproducibility, we will control the seed of random generators:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    result_root = '/home/aa35037123/Wesley/ai_capstone/project1/result'
    dataset_root = '/home/aa35037123/Wesley/ai_capstone/dataset/vehicle_split'
    ckpt_root = '/home/aa35037123/Wesley/ai_capstone/project1/ckpt'
    train_dataset = VehicleDataset(dataset_root, mode='train')
    val_dataset = VehicleDataset(dataset_root, mode='val')
    test_dataset = VehicleDataset(dataset_root, mode='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    print(f'len of train_dataset: {len(train_dataset)}')
    print(f'len of val_dataset: {len(val_dataset)}')
    print(f'len of test_dataset: {len(test_dataset)}')  
    # print(model) # print the model structure
    # Load pretrain weight if ckpt is provided
    if args.ckpt_model is not None:
        model = mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        print(f'Load model from ckpt: {args.ckpt_model}')
        model_ckpt_path = os.path.join(ckpt_root, args.ckpt_model) 
        model.load_state_dict(torch.load(model_ckpt_path))
    else:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        print(f'Use mobilenet_v2 pretrain weight')
        # Change the last layer to fit the number of classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.cuda()
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # Send the model from cpu to gpu
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if(args.ckpt_opt is not None):
        print(f'Load optimizer from ckpt: {args.ckpt_opt}')
        opt_ckpt_path = os.path.join(ckpt_root, args.ckpt_opt) 
        optimizer.load_state_dict(torch.load(opt_ckpt_path))
    print(f'########## Mode: {args.mode} ##########')
    if(args.mode == 'train'):
        Trainer(args, model, train_loader, val_loader, criterion, optimizer, result_root, ckpt_root)
    else:
        evaluate(model, test_loader, result_root)