import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms

# Cifar10 or Cifar100
# dataset = datasets.CIFAR100(
#     root="/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/",
#     train=True,
# )

# images = []
# for image, label in dataset:
#     image = np.asarray(image)
#     images.append(image)

# images = np.stack(images).astype(np.uint8)
# print(images.shape)
# np.save("/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/cifar100.npy", images)

# STL10
dataset = datasets.STL10(
    root='/afs/crc.nd.edu/user/d/dzeng2/data/stl10/', 
    split='unlabeled',
    download=False,
    transform=transforms.Compose([
        transforms.Resize(64),
    ]),
)
# If only use a subset
dataset = torch.utils.data.Subset(dataset, list(range(0, 10000, 1)))

images = []
for image, label in dataset:
    image = np.asarray(image)
    images.append(image)

images = np.stack(images).astype(np.uint8)
print(images.shape)
np.save("/afs/crc.nd.edu/user/d/dzeng2/data/stl10/stl10_subset10000.npy", images)
