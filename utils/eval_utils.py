import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.utils import *


def get_data_loaders(config, batch_size=256):

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }

    if config["dataset"] == "cifar10":
        train_dataset = datasets.CIFAR10(
                root=config["dataset_path"],
                train=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
                ]
            )
        )
        test_dataset = datasets.CIFAR10(
                root=config["dataset_path"],
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
                ]
            )
        )
    elif config["dataset"] == "cifar100":
        train_dataset = datasets.CIFAR100(
                root=config["dataset_path"],
                train=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
                ]
            )
        )
        test_dataset = datasets.CIFAR100(
                root=config["dataset_path"],
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
                ]
            )
        )
    elif config["dataset"] == "stl10":
        train_dataset = datasets.STL10(
                root=config["dataset_path"],
                split='train', 
                transform=transforms.Compose([
                    transforms.Resize(size=64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
                ]
            ),
        )
        test_dataset = datasets.STL10(
                root=config["dataset_path"],
                split='test', 
                transform=transforms.Compose([
                    transforms.Resize(size=64),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
                ]
            ),
        )
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)

    return train_loader, test_loader


def linear_eval(backbone, state_dict, config, device='cuda:0', use_dp=False):
    # define model.
    if config['dataset'] in ['cifar10']:
        num_classes = 10
        model = get_resnet(backbone=backbone, size='small', head='linear', feat_dim=num_classes)
    elif config['dataset'] == 'cifar100':
        num_classes = 100
        model = get_resnet(backbone=backbone, size='small', head='linear', feat_dim=num_classes)
    elif config['dataset'] == 'stl10':
        num_classes = 10
        model = get_resnet(backbone=backbone, size='mid', head='linear', feat_dim=num_classes)

    # ignore the fc in the saved state_dict.
    for k in list(state_dict.keys()):
        if k.startswith('fc'):
            del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    in_features = model.fc.in_features
    model.fc = torch.nn.Identity()
    classifier = torch.nn.Linear(in_features, num_classes)
    if use_dp:
        model = torch.nn.DataParallel(model)
    model.to(device)
    classifier.to(device)
    train_loader, test_loader = get_data_loaders(config=config, batch_size=256)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=config["linear_eval_lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # By default, the linear evaluation runs for 100 epochs.
    epochs = 100
    model.eval()
    for epoch in range(epochs):
        train_top1 = AverageMeter()
        classifier.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            features = model(x_batch)
            logits = classifier(features.detach())
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            train_top1.update(top1[0].item(), x_batch.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_top1 = AverageMeter()
        test_top5 = AverageMeter()
        classifier.eval()
        with torch.no_grad():
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                features = model(x_batch)
                logits = classifier(features)
                top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                test_top1.update(top1[0].item(), x_batch.shape[0])
                test_top5.update(top5[0].item(), x_batch.shape[0])
        
        print(f"Linear eval\tEpoch {epoch+1}/{epochs}\tTop1 Train accuracy {train_top1.avg:.2f}\tTop1 Test accuracy: {test_top1.avg:.2f}\tTop5 test acc: {test_top5.avg:2f}")

    return test_top1.avg, test_top5.avg