"""Train diffusion model."""

import os
import copy
import torch
import yaml
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from models.unet import UNet
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from ddpm import GaussianDiffusionSampler, GaussianDiffusionTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from torch.utils.tensorboard import SummaryWriter


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_data(rank, world_size, config):
    # dataset
    # data transform follows the implementation https://github.com/w86763777/pytorch-ddpm/blob/master/main.py#L97
    if config['dataset'] == 'cifar10':
        dataset = datasets.CIFAR10(
            root=config['data_path'],
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    elif config['dataset'] == 'cifar100':
        dataset = datasets.CIFAR100(
            root=config['data_path'],
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    elif config['dataset'] == 'stl10':
        dataset = datasets.STL10(
            root=config['data_path'], 
            split='unlabeled', 
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    else:
        raise NotImplementedError(f"dataset {config['dataset']} does not support")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=0, drop_last=True, pin_memory=False, sampler=sampler)
    return dataloader


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def warmup_lr(step):
    # warmup = 3200
    warmup = 5000
    return min(step, warmup) / warmup


def train(rank, world_size, config):
    """Train with distributed data parallel.
    
    Check this tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case
    """
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)
    dataloader = prepare_data(rank, world_size, config)
    
    # tensorboard summary
    writer = None
    if rank % world_size == 0:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        summary_string = f"uncondition_diffusion_{config['dataset']}_{current_time}"
        writer = SummaryWriter(f"runs/{summary_string}")
        save_weight_dir = os.path.join(f"./output/{summary_string}", config["save_weight_dir"])
        sampled_dir = os.path.join(f"./output/{summary_string}", config["sampled_dir"])
        os.makedirs(save_weight_dir, exist_ok=True)
        os.makedirs(sampled_dir, exist_ok=True)
        # save args to file.
        with open(os.path.join(f"./output/{summary_string}", 'config.txt'), 'w') as f:
            for eachArg, value in config.items():
                f.writelines(eachArg + ': ' + str(value) + '\n')
    # model setup
    net_model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"], num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(rank)
    if rank % world_size == 0:
        ema_model = copy.deepcopy(net_model)
        ema_sampler = GaussianDiffusionSampler(ema_model, config["beta_1"], config["beta_T"], config["T"]).to(rank)
    trainer = GaussianDiffusionTrainer(net_model, config["beta_1"], config["beta_T"], config["T"]).to(rank)
    ddp_trainer = DDP(trainer, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(ddp_trainer.parameters(), lr=config["lr"])
    warmUpScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lr)

    # start training
    n_iter = 0
    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            ddp_trainer.train()
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x = images.to(rank)
                loss = ddp_trainer(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_trainer.parameters(), config["grad_clip"])
                optimizer.step()
                warmUpScheduler.step()
                if rank % world_size == 0:
                    # EMA decay rate: https://github.com/w86763777/pytorch-ddpm/blob/master/main.py#L42C33-L42C39
                    ema(net_model, ema_model, 0.9999)
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img shape": x.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                if writer and n_iter % 100 == 0:
                    writer.add_scalar('loss', loss.item(), global_step=n_iter)
                n_iter += 1

        # warmUpScheduler.step()
        if writer:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=e)
        
        # save and eval only happens in the one process (rank==0)
        if (e+1) % config["save_ckpt_interval"] == 0 and rank % world_size == 0:
            # save the training model
            torch.save(ddp_trainer.module.model.state_dict(), os.path.join(save_weight_dir, 'ckpt_' + str(e) + "_.pt"))
            # save the ema model
            torch.save(ema_model.state_dict(), os.path.join(save_weight_dir, 'ckpt_ema_' + str(e) + "_.pt"))
            # Do one sampling each time we save the model.
            with torch.no_grad():
                # Sampled from standard normal distribution
                noisyImage = torch.randn(size=[config["batch_size"], 3, config["img_size"], config["img_size"]], device=rank)
                sampledImgs = ema_sampler(noisyImage)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                save_image(sampledImgs, os.path.join(sampled_dir, config["sampledImgName"]+f'_{e:03d}.png'), nrow=config["nrow"])

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_cifar10.yaml")
    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    mp.spawn(train, args=(world_size, config,), nprocs=world_size, join=True)
