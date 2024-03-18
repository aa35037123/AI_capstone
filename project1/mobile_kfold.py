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
from dataloader_merge import VehicleDataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import time
import os
from sklearn.model_selection import StratifiedKFold
from glob import glob
from torchvision import transforms

BATCH_SIZE = 10
LEARNING_RATE = 0.0001
NUM_EPOCH = 10
NUM_CLASSES = 7
# TRANSFORMS = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

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

def evaluate(model, val_loader, result_root, global_confusion_matrix):
    model.eval()
    num_samples = 0
    num_corrects = 0
    for images, labels in tqdm(val_loader, desc='Evaluating'):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicts = torch.max(outputs, dim=1)
        num_samples += labels.size(0)
        num_corrects += (predicts == labels).sum()
        for i in range(labels.size(0)):
            global_confusion_matrix[labels[i]][predicts[i]] += 1
    accuracy = num_corrects / num_samples
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy, global_confusion_matrix

def plot_confusion(result_root, global_confusion_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(global_confusion_matrix, annot=True, fmt="d", cmap='Blues', 
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Kfold Confusion Matrix')
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=0)
    plt.tight_layout()  # Adjust the layout
    plt.savefig(os.path.join(result_root, 'final_confusion_matrix_resample.jpg'))
    plt.close()
    
def plot_class_distribution(original_sizes, resampled_sizes, title, result_root):
    classes = list(original_sizes.keys())
    n_classes = len(classes)
    index = np.arange(n_classes)
    bar_width = 0.35
    
    plt.figure(figsize=(12, 8))
    plt.bar(index, list(original_sizes.values()), bar_width, label='Original')
    plt.bar(index + bar_width, list(resampled_sizes.values()), bar_width, label='Resampled')

    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.xticks(index + bar_width / 2, classes)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_root, f'data_distribution.jpg'))
    plt.close()
# updata each class to have the same number of samples
                
def get_kfold_resampling(dataset_root, result_root):
    classes = sorted(os.listdir(dataset_root))
    img_paths = []
    img_labels = []
    indices_per_class = {cls: [] for cls in classes} # each class has a list of indices

    for cls_name in classes:
        cls_index = classes.index(cls_name)
        cls_path = os.path.join(dataset_root, cls_name)
        for img_path in glob(os.path.join(cls_path, '*.jpg')):
            img_paths.append(img_path)
            img_labels.append(cls_index)
            indices_per_class[cls_name].append(len(img_paths) - 1) # append the index of the last element
    
    original_class_size = {cls:len(indices) for cls, indices in indices_per_class.items()}
    min_class_size = min(len(indices) for indices in indices_per_class.values()) # find the min length for all classes

    downsampled_indices = []
    for cls_indices in indices_per_class.values():
        downsampled_indices.extend(random.sample(cls_indices, min_class_size)) # randomly sample the same number of indices from each class
    
    resampled_class_size = {cls:min_class_size for cls in classes}
    img_paths = [img_paths[i] for i in downsampled_indices]
    img_labels = [img_labels[i] for i in downsampled_indices]
    
    plot_class_distribution(original_class_size, resampled_class_size, 'Class Distribution', result_root)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = kf.split(img_paths, img_labels)  # This will be used for cross-validation
    
    return splits, img_paths, img_labels

def get_kfold(dataset_root):
    classes = sorted(os.listdir(dataset_root))
    img_paths = []
    img_labels = []
    for cls_name in classes:
        cls_index = classes.index(cls_name)
        cls_path = os.path.join(dataset_root, cls_name)
        for img_path in glob(os.path.join(cls_path, '*.jpg')):
            img_paths.append(img_path)
            img_labels.append(cls_index)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = kf.split(img_paths, img_labels)  # This will be used for cross-validation
    
    return splits, img_paths, img_labels

def Trainer(
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        result_root: str,
        ckpt_root: str,
        fold: int
) -> float: 
    # Send the model from cpu to gpu
    best_acc = 0
    best_model = None
    
    for epoch in tqdm(range(1, NUM_EPOCH+1)):
        train(model, train_loader, criterion, optimizer)
        # acc = evaluate(model, val_loader, result_root, fold)
        # print(f'epoch, {epoch}, accuracy: {acc:.4f}')
        
        # if acc > best_acc:
        #     best_acc = acc
        #     best_model = model.state_dict()

    # print(f'Finished Training one fold, best accuracy: {best_acc:.4f}, learning rate: {LEARNING_RATE}')
    
    
    return best_acc, best_model


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
    result_root = '/home/aa35037123/Wesley/ai_capstone/project1/result/mobilenet'
    dataset_root = '/home/aa35037123/Wesley/ai_capstone/dataset/vehicle_merged'
    ckpt_root = '/home/aa35037123/Wesley/ai_capstone/project1/ckpt'
    
    print(f'########## Mode: {args.mode} ##########')
    # kfold_split, img_paths, img_labels = get_kfold(dataset_root)
    kfold_split, img_paths, img_labels = get_kfold_resampling(dataset_root, result_root)
    kfold_accs = []
    best_kfold_acc = 0
    best_kfold_model = None
    # Initial confusion matrix
    global_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(kfold_split):
        print(f'Fold: {fold}')
        # generate fold dataset
        train_dataset = VehicleDataset([img_paths[i] for i in train_idx], [img_labels[i] for i in train_idx], transform=TRANSFORMS)
        val_dataset = VehicleDataset([img_paths[i] for i in val_idx], [img_labels[i] for i in val_idx], transform=TRANSFORMS)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
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

        best_acc, best_model = Trainer(args, model, train_loader, val_loader, criterion, optimizer, result_root, ckpt_root, fold)
        
        acc, global_confusion_matrix = evaluate(model, val_loader, result_root, global_confusion_matrix)
        print(f'Fold{fold}: accuracy: {acc:.4f}')
        kfold_accs.append(acc)
        if acc > best_kfold_acc:
            best_kfold_acc = acc
            best_kfold_model = model.state_dict()
    plot_confusion(result_root, global_confusion_matrix, os.listdir(dataset_root))
    avg_acc = sum(kfold_accs) / len(kfold_accs)
    print(f'### Finished all Training, avg accuracy: {avg_acc:.4f}, learning rate: {LEARNING_RATE}')
    print(f'all accuracy: {kfold_accs}')
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Save the best model
    # ckpt_name = f'/home/aa35037123/Wesley/ai_capstone/project1/ckpt/model_{timestamp}.ckpt'
    model_ckpt_path = os.path.join(ckpt_root, f'model_{timestamp}.pt')
    torch.save(best_model, model_ckpt_path)
    opt_ckpt_path = os.path.join(ckpt_root, f'opt_{timestamp}.pt')
    torch.save(optimizer.state_dict(), opt_ckpt_path)
    print(f'Save model to {model_ckpt_path}')
    print(f'Save optimizer to {opt_ckpt_path}')
