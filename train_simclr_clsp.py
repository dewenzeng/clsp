"""SimCLR with CLSP"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from datetime import datetime
from pytorch_lightning import seed_everything
from utils.utils import *
from utils.eval_utils import linear_eval
from utils.cos_lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


def similarity_loss_fn(x, y):
    """Similarity loss function, features are already normalized."""
    return 2 - 2 * (x * y).sum(dim=-1)


def info_nce_loss_with_label(features, temperature=0.1, labels=None, mask=None):
    """Supervsed info_nce loss.
    
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
    """
    
    device = features.device
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -1.0 * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def get_dataset(config):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS_N_STD[config['dataset']][0], std=MEANS_N_STD[config['dataset']][1]),
            ]) for _ in range(3)
        ]
        dataset = customized_stl10_dataset(
            root='/afs/crc.nd.edu/user/d/dzeng2/data/stl10',
            split='unlabeled',
            download=True,
            transform=ContrastiveLearningViewGenerator(transform_list),
            synthetic_data_path=config["synthetic_data_path"],
            num_candidates=config["num_candidates"],
        )
    return dataset


def train(config):
    device = torch.device(config["device"])
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], drop_last=True, pin_memory=False)
    
    # tensorboard summary
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_string = f"simclr_clsp_{config['dataset']}_{config['backbone']}_bs-{config['batch_size']}_temp-{str(config['temperature'])}_{current_time}"
    print(f"Output dir: {summary_string}")
    summary_dir = f"runs/{summary_string}"
    writer = SummaryWriter(summary_dir)
    save_weight_dir = os.path.join(f"./output/{summary_string}", config["save_weight_dir"])
    writer.add_text('config', str(config), 0)
    os.makedirs(save_weight_dir, exist_ok=True)

    # save args to file.
    with open(os.path.join(f"./output/{summary_string}", 'config.txt'), 'w') as f:
        for eachArg, value in config.items():
            f.writelines(eachArg + ': ' + str(value) + '\n')

    # model setup
    if config['dataset'] in ['cifar10', 'cifar100']:
        model = get_resnet(backbone=config["backbone"], size='small', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
    elif config['dataset'] in ['stl10']:
        model = get_resnet(backbone=config["backbone"], size='mid', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
    else:
        model = get_resnet(backbone=config["backbone"], size='big', head='mlp', feat_dim=config["feat_dim"], hidden_dim=config["hidden_dim"])
    if config["use_dp"]:
        model = nn.DataParallel(model)
    model.to(device)

    # Use SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])
    warmUpScheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=config["epoch"], warmup_start_lr=config["warmup_start_lr"], eta_min=0.0)

    
    # start training
    n_iter = 0
    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            model.train()
            for images, synthetic_images, labels in tqdmDataLoader:
                # train
                bsz = images[0].shape[0]
                images = torch.cat([images[0], images[1], synthetic_images], dim=0).to(device)
                features = model(images)
                features = torch.nn.functional.normalize(features, dim=1)
                f1, f2, f3 = torch.split(features, [bsz, bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_simclr = info_nce_loss_with_label(features, temperature=config['temperature'])
                loss_clsp =  similarity_loss_fn(f1, f3).mean()
                loss = loss_simclr + config["clsp_loss_weight"] * loss_clsp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "Train epoch": e,
                    "loss": loss.item(),
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                if n_iter % config["log_interval"] == 0:
                    writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    writer.add_scalar('loss_simclr', loss_simclr.item(), global_step=n_iter)
                    writer.add_scalar('loss_clsp', loss_clsp.item(), global_step=n_iter)
                n_iter += 1

        warmUpScheduler.step()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=e)
        if (e+1) % config["save_ckpt_interval"] == 0:
            torch.save(model.module.state_dict() if config["use_dp"] else model.state_dict(), os.path.join(
                save_weight_dir, 'ckpt_' + str(e) + ".pt"))
            # Do one evaluation.
            top1, top5 = linear_eval(
                backbone=config["backbone"],
                state_dict=model.module.state_dict() if config["use_dp"] else model.state_dict(), 
                config=config,
                device=device
            )
            writer.add_scalar('test/top1', top1, global_step=e+1)
            writer.add_scalar('test/top5', top5, global_step=e+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/simclr_clsp_cifar10.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    seed_everything(config["seed"])
    train(config)
