from typing import Dict, Optional, Callable, Tuple, Any

import os
import torch
import pickle
import torchvision
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torchvision import datasets
from models.resnet_sk import get_resnet_sk


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.
        Args:
            img (Image): an image in the PIL.Image format.
        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _cosine_simililarity(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, N, C)
    # v shape: (N, N)
    v = torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))
    return v


class ContrastiveLearningViewGenerator(object):
    """Generate a multiple views of the same image."""

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x):
        return [transform(x) for transform in self.transform_list]


def get_resnet(backbone='resnet18', size='small', head='mlp', feat_dim=256, hidden_dim=2048):
    if backbone == 'resnet18':
        model = torchvision.models.resnet18()
    elif backbone == 'resnet50':
        model = torchvision.models.resnet50()
    elif backbone == 'resnet101':
        model = torchvision.models.resnet101()
    elif backbone == 'resnet101_sk':
        model, _ = get_resnet_sk(depth=101, width_multiplier=1, sk_ratio=0.0625)
    input_features = model.fc.in_features
    if hidden_dim is None:
        hidden_dim = input_features
    if head == 'mlp':
        model.fc = nn.Sequential(
                nn.Linear(input_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feat_dim)
            )
    elif head == 'linear':
        model.fc = nn.Linear(input_features, feat_dim)
    if backbone == 'resnet101_sk':
        return model
    # For small resolution dataset like cifar10 and cifar100.
    if size == 'small':
       model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
       model.maxpool = nn.Identity()
    elif size == 'mid':
       # https://github.com/htdt/self-supervised/blob/master/model.py#L22
       model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

    
class customized_cifar10_dataset(datasets.CIFAR10):
    """Add additional index to the get function."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        synthetic_data_path: str = "synthetic_10.npy",
        num_candidates: int = 1,
        transform2: bool = None,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        # Set transform2 if you want to use a different transform for the synthetic images.
        self.transform2 = transform2
        self.synthetic_data_path = synthetic_data_path
        self.num_candidates = num_candidates
        if synthetic_data_path:
            self.synthetic_images = np.load(synthetic_data_path)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, image_ids) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        # If there is any generated images of anchor with index, use it. Otherwise, the selected image is just another data augmentation.
        if self.synthetic_data_path:
            num_candidates = min(self.num_candidates, self.synthetic_images.shape[1])
            selected_idx = np.random.randint(0, num_candidates)
            selected_image = self.synthetic_images[index, selected_idx]
            selected_image = Image.fromarray(selected_image)
            if self.transform is not None:
                selected_image = self.transform2(selected_image) if self.transform2 else self.transform(selected_image)
            selected_image = selected_image[0]
        else:
            selected_image = image[-1]
        return image, selected_image, target


class customized_cifar100_dataset(datasets.CIFAR100):
    """Add additional index to the get function."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        synthetic_data_path: str = "synthetic_10.npy",
        num_candidates: int = 1,
        transform2: bool = None,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.transform2 = transform2
        self.synthetic_data_path = synthetic_data_path
        self.num_candidates = num_candidates
        if synthetic_data_path:
            self.synthetic_images = np.load(synthetic_data_path)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, image_ids) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        # If there is any generated images of anchor with index, use it. Otherwise, the selected image is just another data augmentation.
        if self.synthetic_data_path:
            num_candidates = min(self.num_candidates, self.synthetic_images.shape[1])
            selected_idx = np.random.randint(0, num_candidates)
            selected_image = self.synthetic_images[index, selected_idx]
            selected_image = Image.fromarray(selected_image)
            if self.transform is not None:
                selected_image = self.transform2(selected_image) if self.transform2 else self.transform(selected_image)
            selected_image = selected_image[0]
        else:
            selected_image = image[-1]
        return image, selected_image, target


class customized_stl10_dataset(datasets.STL10):
    """Add additional index to the get function."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        synthetic_data_path: str = "synthetic_10",
        num_candidates: int = 1,
        transform2: bool = None,
    ) -> None:
        super().__init__(root, split, folds, transform, target_transform, download)
        self.transform2 = transform2
        self.synthetic_data_path = synthetic_data_path
        self.num_candidates = num_candidates
        if synthetic_data_path:
            self.synthetic_images = np.load(synthetic_data_path)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, image_ids) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        # If there is any generated images of anchor with index, use it. Otherwise, the selected image is just another data augmentation.
        if self.synthetic_data_path:
            num_candidates = min(self.num_candidates, self.synthetic_images.shape[1])
            selected_idx = np.random.randint(0, num_candidates)
            selected_image = self.synthetic_images[index, selected_idx]
            selected_image = Image.fromarray(selected_image)
            if self.transform is not None:
                selected_image = self.transform2(selected_image) if self.transform2 else self.transform(selected_image)
            selected_image = selected_image[0]
        else:
            selected_image = image[-1]
        return image, selected_image, target

