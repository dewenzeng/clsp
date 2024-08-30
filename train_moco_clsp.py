"""MoCo V2"""

import os
import random
import torch
import math
import yaml
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from PIL import ImageFilter
from utils.utils import *
from utils.eval_utils import linear_eval
from utils.cos_lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

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

def similarity_loss_fn(x, y):
    """Similarity loss function, features are already normalized."""
    return 2 - 2 * (x * y).sum(dim=-1)

class MoCo(nn.Module):
    """
    Code copied from https://github.com/facebookresearch/moco/blob/main/moco/builder.py, ddp related code is removed.
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder_q, encoder_k, dim=128, K=65536, m=0.999):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # keep the two augs queue, https://github.com/vturrisi/solo-learn/blob/main/solo/methods/mocov2plus.py#L68
        self.register_buffer("queue", torch.randn(2, dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """keys: [n_view, b, d]"""
        batch_size = keys.shape[1]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)    # [n_view, d, b]
        self.queue[:, :, ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def update_tau(self, cur_step, max_steps):
        # final tau and base tau from https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/mocov2plus.yaml#L23C9-L23C10
        cur_tau = (
            0.999
            - (0.999 - 0.99) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )
        self.m = cur_tau

    def forward(self, im_q, im_k, im_a):
        """
        Check symetric forward https://github.com/vturrisi/solo-learn/blob/main/solo/methods/mocov2plus.py#L179

        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            im_a: a batch of additional positives
        Output:
            logits, targets
        """

        # compute query features
        q1 = self.encoder_q(im_q)  # queries: NxC
        q2 = self.encoder_q(im_k)  # queries: NxC
        q3 = self.encoder_q(im_a)  # queries: NxC
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q3 = nn.functional.normalize(q3, dim=1)

        with torch.no_grad():
            k1 = self.encoder_k(im_q)  # keys: NxC
            k2 = self.encoder_k(im_k)  # keys: NxC
            k3 = self.encoder_k(im_a)  # keys: NxC
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k3 = nn.functional.normalize(k3, dim=1)

        return q1, q2, q3, k1, k2, k3

def mocov2plus_loss_func(
    query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor, temperature=0.1
) -> torch.Tensor:
    """Computes MoCo's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the keys from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): temperature of the softmax in the contrastive
            loss. Defaults to 0.1.

    Returns:
        torch.Tensor: MoCo loss.
    """

    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    neg = torch.einsum("nc,ck->nk", [query, queue])

    # logits
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)

    return F.cross_entropy(logits, targets)


def get_dataset(config):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    MEANS_N_STD = {
            "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            "imagenet100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        }
    if config['dataset'] == 'cifar10':
        transform_list = [
            transforms.Compose([
                transforms.RandomResizedCrop(size=config["img_size"], scale=(0.08, 1.0)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
            ]) for _ in range(3)
        ]
        dataset = customized_cifar10_dataset(
            root=config["dataset_path"],
            train=True, download=True,
            transform=ContrastiveLearningViewGenerator(transform_list),
            synthetic_data_path=config["synthetic_data_path"],
            num_candidates=config["num_candidates"],
        )
    elif config['dataset'] == 'cifar100':
        transform_list = [
            transforms.Compose([
                transforms.RandomResizedCrop(size=config["img_size"], scale=(0.08, 1.0)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
            ]) for _ in range(3)
        ]
        dataset = customized_cifar100_dataset(
            root=config["dataset_path"],
            train=True, download=True,
            transform=ContrastiveLearningViewGenerator(transform_list),
            synthetic_data_path=config["synthetic_data_path"],
            num_candidates=config["num_candidates"],
        )
    elif config['dataset'] == 'stl10':
        transform_list = [
            transforms.Compose([
                transforms.RandomResizedCrop(size=config["img_size"]),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
            ]) for _ in range(3)
        ]
        dataset = customized_stl10_dataset(
            root=config["dataset_path"],
            split='unlabeled',
            download=True,
            transform=ContrastiveLearningViewGenerator(transform_list),
            synthetic_data_path=config["synthetic_data_path"],
            num_candidates=config["num_candidates"],
        )
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented.")
    return dataset

def train(config: Dict):
    device = torch.device(config["device"])
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], drop_last=True, pin_memory=False)
    
    # tensorboard summary
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_string = f"moco_clsp_{config['dataset']}_{config['backbone']}_bs-{config['batch_size']}_temp-{str(config['temperature'])}_{current_time}"
    print(f"Output dir: {summary_string}")
    summary_dir = f"runs/{summary_string}"
    writer = SummaryWriter(summary_dir)
    writer.add_text('config', str(config), 0)
    save_weight_dir = os.path.join(f"./output/{summary_string}", config["save_weight_dir"])
    writer.add_text('config', str(config), 0)
    os.makedirs(save_weight_dir, exist_ok=True)

    # save args to file.
    with open(os.path.join(f"./output/{summary_string}", 'config.txt'), 'w') as f:
        for eachArg, value in config.items():
            f.writelines(eachArg + ': ' + str(value) + '\n')

    # model setup
    if config['dataset'] in ['cifar10', 'cifar100']:
        encoder_q = get_resnet(backbone=config["backbone"], size='small', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
        encoder_k = get_resnet(backbone=config["backbone"], size='small', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
    elif config['dataset'] in ['stl10']:
        encoder_q = get_resnet(backbone=config["backbone"], size='mid', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
        encoder_k = get_resnet(backbone=config["backbone"], size='mid', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
    else:
        encoder_q = get_resnet(backbone=config["backbone"], size='big', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
        encoder_k = get_resnet(backbone=config["backbone"], size='big', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
    
    model = MoCo(
        encoder_q=encoder_q,
        encoder_k=encoder_k,
        dim=config["feat_dim"],
        K=config["moco_K"],
        m=config["moco_momentum"],
    )
    
    if config["use_dp"]:
        model = nn.DataParallel(model)
    model.to(device)

    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    warmUpScheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=config["epoch"], warmup_start_lr=config["warmup_start_lr"], eta_min=0.0)

    # start training
    n_iter = 0
    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            model.train()
            for images, synthetic_images, labels in tqdmDataLoader:
                # train
                images[0] = images[0].to(device, non_blocking=True)
                images[1] = images[1].to(device, non_blocking=True)
                image_add = synthetic_images.to(device, non_blocking=True)
                # update EMA encoder.
                model.module.momentum_update_key_encoder() if config["use_dp"] else model.momentum_update_key_encoder()
                q1, q2, q3, k1, k2, k3 = model(images[0], images[1], image_add)
                queue = model.module.queue.clone().detach() if config["use_dp"] else model.queue.clone().detach()
                loss_moco = (mocov2plus_loss_func(q1, k2, queue[1], config["temperature"]) + mocov2plus_loss_func(q2, k1, queue[0], config["temperature"])) / 2
                # update memory bank.
                keys = torch.stack((k1, k2))  # [2, B, D]
                model.module.dequeue_and_enqueue(keys) if config["use_dp"] else model.dequeue_and_enqueue(keys)
                loss_clsp = similarity_loss_fn(q1, q3).mean()
                loss = loss_moco + config["clsp_loss_weight"] * loss_clsp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update momentum tau.
                if config["use_dp"]:
                    model.module.update_tau(
                        cur_step=n_iter,
                        max_steps=config["epoch"] * len(dataloader),
                    )
                else:
                    model.update_tau(
                        cur_step=n_iter,
                        max_steps=config["epoch"] * len(dataloader),
                    )
                tqdmDataLoader.set_postfix(ordered_dict={
                    "Train epoch": e,
                    "loss": loss.item(),
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                if n_iter % config["log_interval"] == 0:
                    writer.add_scalar('loss_moco', loss_moco.item(), global_step=n_iter)
                    writer.add_scalar('loss_clsp', loss_clsp.item(), global_step=n_iter)
                    writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    writer.add_scalar('momentum_tau', model.module.m if config["use_dp"] else model.m, global_step=n_iter)
                n_iter += 1

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=e)
        warmUpScheduler.step()
        if (e+1) % config["save_ckpt_interval"] == 0:
            # Save checkpoint.
            torch.save(model.module.encoder_q.state_dict() if config["use_dp"] else model.encoder_q.state_dict(), os.path.join(
                save_weight_dir, 'ckpt_' + str(e) + ".pt"))
            # Do one evaluation.
            top1, top5 = linear_eval(
                backbone=config["backbone"],
                state_dict=model.module.encoder_q.state_dict() if config["use_dp"] else model.encoder_q.state_dict(), 
                config=config,
                device=device
            )
            writer.add_scalar('test/top1', top1, global_step=e+1)
            writer.add_scalar('test/top5', top5, global_step=e+1)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/moco_clsp_cifar10.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    seed_everything(config["seed"])
    train(config)
