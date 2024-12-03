import argparse
import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # tqdm 라이브러리 추가

def get_data_transforms(dataset_name):
    if dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761]),
        ])
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError('Unsupported dataset')
    return transform

def get_datasets(dataset_name, data_dir):
    transform = get_data_transforms(dataset_name)
    if dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'imagenet':
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        num_classes = 1000
    else:
        raise ValueError('Unsupported dataset')
    return train_dataset, val_dataset, num_classes

def get_model(num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f'Epoch [{epoch}/{total_epochs}] Training', leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, desc='Validation', leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='ViT Training on CIFAR-100 or ImageNet')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'],
                        help='Dataset to use (cifar100 or imagenet)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory where data is stored')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # 데이터셋 및 데이터로더 설정
    train_dataset, val_dataset, num_classes = get_datasets(args.dataset, args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 모델, 손실 함수, 옵티마이저 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f'Epoch [{epoch}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Val Loss: {val_loss:.4f} '
              f'Val Accuracy: {val_accuracy:.2f}%')

    # 모델 저장
    model_save_path = f'vit_{args.dataset}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    main()

