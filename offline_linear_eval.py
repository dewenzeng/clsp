import torch
from utils.eval_utils import linear_eval

# Step 1: setup the config
# Cifar10
config = {
    "dataset": "cifar10",
    "dataset_path": "/afs/crc.nd.edu/user/d/dzeng2/data/cifar10/",
    "backbone": "resnet50",
    "device": "cuda:0",
    "state_dict": "./pretrained_ckpts/simclr_clsp_cifar10_resnet50.pt",
    "linear_eval_lr": 1.0,
    "use_dp": True,
}
# Cifar100
# config = {
#     "dataset": "cifar100",
#     "dataset_path": "/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/",
#     "backbone": "resnet18",
#     "device": "cuda:0",
#     # "state_dict": "/afs/crc.nd.edu/user/d/dzeng2/code/clsp/output/simclr_clsp_cifar100_resnet18_bs-1024_temp-0.2_2024-08-27_22-25-06/ckpts/ckpt_999.pt",
#     "state_dict": "./pretrained_ckpts/moco_cifar100_resnet18.pt",
#     "linear_eval_lr": 1.0,
#     "use_dp": False,
# }

# Step 2: load pre-trained state dict
state_dict = torch.load(config["state_dict"])

# Step 3: start linear evaluation
top1, top5 = linear_eval(
    backbone=config["backbone"],
    state_dict=state_dict, 
    config=config,
    device=config["device"],
    use_dp=config["use_dp"]
)

print(f"Final test top1:{top1:.2f}, top5:{top5:.2f}")
