"""Generate synthetic dataset using pretrained diffusion model."""

import os
import yaml
from typing import Dict, Optional, Callable, Tuple, Any
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from ddpm import GaussianDiffusionSampler
from models.unet import UNet


class customized_cifar10_dataset(datasets.CIFAR10):
    """Add additional index to the get function."""
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, image_ids) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        return image, target, index
    
class customized_cifar100_dataset(datasets.CIFAR100):
    """Add additional index to the get function."""
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, image_ids) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        return image, target, index
    
class customized_stl10_dataset(datasets.STL10):
    """Add additional index to the get function."""
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, image_ids) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        return image, target, index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--ckpt_path', default="", type=str, help='the ckpt path for pretrained diffusion model')
    parser.add_argument("--config", type=str, default="configs/diffusion_cifar10.yaml", help="config file for pretrain the diffusion model")
    parser.add_argument('--save_dir', default="", type=str, help='the path to save the generated file')
    parser.add_argument('--num_candidates', default=0, type=int, help='the number of synthetic images generated for each sample in the dataset')
    parser.add_argument('--batch_size', default=256, type=int, help='the batch size used for diffusion sampling')
    parser.add_argument('--interpolation_weight', default=0.0, type=float, help='interpolation weight')
    parser.add_argument('--ddim_sampling_timesteps', default=100, type=int, help='ddim sampling timesteps')
    parser.add_argument('--ddim_eta', default=1.0, type=float, help='ddim ddim_eta')
    parser.add_argument('--sample_method', default='ddpm', type=str, help='diffusion sample method', choices=['ddpm', 'ddpm_interpolation', 'ddim', 'ddim_interpolation'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['dataset'] =='cifar10':
        dataset = customized_cifar10_dataset(
            root=config['data_path'], 
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            download=False,
        )
    elif config['dataset'] =='cifar100':
        dataset = customized_cifar100_dataset(
            root=config['data_path'], 
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            download=False,
        )
    elif config['dataset'] =='stl10':
        dataset = customized_stl10_dataset(
            root='/afs/crc.nd.edu/user/d/dzeng2/data/stl10/', 
            split='unlabeled',
            transform=transforms.Compose([
                transforms.Resize(config['img_size']),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            download=False,
        )
        # If only use a subset
        # dataset = torch.utils.data.Subset(dataset, list(range(0, 10000, 1)))
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not supported.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)
    os.makedirs(args.save_dir, exist_ok=True)
    # create the dataset buffer
    for candidate_idx in range(args.num_candidates):
        generated_images = np.zeros([len(dataset), config["img_size"], config["img_size"], 3], dtype=np.uint8)
        print(f'generating for candidate index: {candidate_idx}')

        # load model and evaluate
        with torch.no_grad():
            device = torch.device(config["device"])
            model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"], num_res_blocks=config["num_res_blocks"], dropout=0.)
            ckpt = torch.load(args.ckpt_path, map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            sampler = GaussianDiffusionSampler(model, config["beta_1"], config["beta_T"], config["T"]).to(device)
            sampler = torch.nn.DataParallel(sampler)
            for batch_idx, (anchor_image, label, idx) in enumerate(dataloader):
                print(f"sample {batch_idx+1}/{len(dataloader)}")
                anchor_image = anchor_image.to(device)
                # Sampled from standard normal distribution.
                noisyImage = torch.randn(size=[anchor_image.shape[0], 3, config["img_size"], config["img_size"]], device=device)
                sampledImgs = sampler(noisyImage, x_anchor=anchor_image, weight=args.interpolation_weight, sample_method=args.sample_method, ddim_sampling_timesteps=args.ddim_sampling_timesteps, ddim_eta=args.ddim_eta)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                # Save generated images.
                for i in range(len(sampledImgs)):
                    for_save = sampledImgs[i].cpu().permute(1,2,0).numpy() * 255
                    generated_images[idx[i].item()] = for_save.astype(np.uint8)

        # save the result for each candidate index.
        np.save(os.path.join(args.save_dir, f'generated_{args.sample_method}_{args.interpolation_weight}_{candidate_idx:03d}.npy'), generated_images)
    
    # combining all synthetic images.
    all_images = []
    for candidate_idx in range(args.num_candidates):
        all_images.append(np.load(os.path.join(args.save_dir, f'generated_{args.sample_method}_{args.interpolation_weight}_{candidate_idx:03d}.npy')))
    all_images = np.stack(all_images, axis=1)
    print(f'all_images:{all_images.shape}')
    np.save(os.path.join(args.save_dir, f"synthetic_{config['dataset']}_{args.sample_method}_{args.interpolation_weight}_{args.num_candidates}_candidates.npy"), all_images)
