import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
from utils.utils import get_resnet
from utils.eval_utils import get_data_loaders
from utils.knn import WeightedKNNClassifier

@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features and labels.
    """

    model.eval()
    features, labels = [], []
    for image, label in tqdm(loader):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        outs = model(image)
        features.append(outs.detach())
        labels.append(label)
    model.train()
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn.update(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute()

    # free up memory
    del knn

    return acc1, acc5


# Step 1: setup the config
# Cifar10
config = {
    "dataset": "cifar100",
    "dataset_path": "/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/",
    "backbone": "resnet50",
    "state_dict": "./pretrained_ckpts/moco_clsp_cifar100_resnet50.pt",
    "k": [5, 10, 20, 50, 100, 200],
    "temperature": 0.07,
    "distance_function": "cosine",
}

state_dict = torch.load(config["state_dict"])

if config['dataset'] in ['cifar10']:
    num_classes = 10
    model = get_resnet(backbone=config['backbone'], size='small', head='linear', feat_dim=num_classes)
elif config['dataset'] == 'cifar100':
    num_classes = 100
    model = get_resnet(backbone=config['backbone'], size='small', head='linear', feat_dim=num_classes)
elif config['dataset'] == 'stl10':
    num_classes = 10
    model = get_resnet(backbone=config['backbone'], size='mid', head='linear', feat_dim=num_classes)
else:
    raise NotImplementedError(f"dataset {config['dataset']} not supported")

# ignore the fc in the saved state_dict.
for k in list(state_dict.keys()):
    if k.startswith('fc'):
        del state_dict[k]

log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']
model.fc = torch.nn.Identity()

model.cuda()

# prepare data
train_loader, test_loader = get_data_loaders(config, batch_size=256)

# extract train features
train_features, train_targets = extract_features(train_loader, model)

# extract test features
test_features, test_targets = extract_features(test_loader, model)

# run k-nn for all possible combinations of parameters
accs = []
for k in config["k"]:
    print("---")
    print(f"Running k-NN with params: distance_fx={config['distance_function']}, k={k}, T={config['temperature']}...")
    acc1, acc5 = run_knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
        k=k,
        T=config['temperature'],
        distance_fx=config['distance_function'],
    )
    accs.append(acc1)
    print(f"result: acc@1={acc1}, acc@5={acc5}")

print(f"Best test top1:{max(accs):.2f}")
