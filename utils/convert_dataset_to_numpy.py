import numpy as np
from torchvision import datasets

dataset = datasets.CIFAR100(
    root="/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/",
    train=True,
)

images = []
for image, label in dataset:
    image = np.asarray(image)
    images.append(image)

images = np.stack(images).astype(np.uint8)
print(images.shape)
np.save("/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/cifar100.npy", images)
