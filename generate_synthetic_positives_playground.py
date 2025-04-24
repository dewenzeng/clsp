import yaml
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision import datasets
from ddpm import GaussianDiffusionSampler
from models.unet import UNet

ckpt_path = "./pretrained_ckpts/uncondition_diffusion_cifar10.pt"
# ckpt_path = "./pretrained_ckpts/uncondition_diffusion_cifar100.pt"
# ckpt_path = "./pretrained_ckpts/uncondition_diffusion_stl10.pt"
config_file = "configs/diffusion_cifar10.yaml"
# config_file = "configs/diffusion_cifar100.yaml"
# config_file = "configs/diffusion_stl10.yaml"
batch_size = 8      # how many positives you want to generate for each sample
interpolation_weight = 0.01
num_interpolation_layers = 1
ddim_sampling_timesteps = 200
ddim_eta = 0.1
sample_method = "ddim_interpolation"
# sample_method = "ddpm_interpolation"
device = "cuda:0"

with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if config['dataset'] =='cifar10':
    dataset = datasets.CIFAR10(
        root=config['data_path'], 
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        download=False,
    )
elif config['dataset'] =='cifar100':
    dataset = datasets.CIFAR100(
        root=config['data_path'], 
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        download=False,
    )
elif config['dataset'] =='stl10':
    dataset = datasets.STL10(
        root=config['data_path'], 
        split='unlabeled',
        transform=transforms.Compose([
            transforms.Resize(config['img_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        download=False,
    )
else:
    raise NotImplementedError(f"Dataset {config['dataset']} not supported.")


device = torch.device(config["device"])
model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"], num_res_blocks=config["num_res_blocks"], dropout=0.)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
print(f"model weight loaded from ckpt {ckpt_path}.")
model.eval()
sampler = GaussianDiffusionSampler(model, config["beta_1"], config["beta_T"], config["T"]).to(device)
sampler = torch.nn.DataParallel(sampler)

# choose a data you want to be the anchor
# data_indices = [202]
data_indices = np.random.randint(low=0, high=10000, size=4).tolist()
images_for_show = []

for data_idx in data_indices:
    anchor_image, _ = dataset[data_idx]
    anchor_images = anchor_image.unsqueeze(dim=0).repeat((batch_size, 1, 1, 1)).to(device)
    # load model and evaluate
    with torch.no_grad():
        # Sampled from standard normal distribution.
        noisyImage = torch.randn(size=[anchor_images.shape[0], 3, config["img_size"], config["img_size"]], device=device)
        sampledImgs = sampler(noisyImage, x_anchor=anchor_images, weight=interpolation_weight, num_interpolation_layers=num_interpolation_layers, sample_method=sample_method, ddim_sampling_timesteps=ddim_sampling_timesteps, ddim_eta=ddim_eta)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        sampledImgs = sampledImgs.cpu()
   
    # save images for visualization.
    anchor_image = anchor_image * 0.5 + 0.5
    images_for_show.append(torch.concatenate([anchor_image.unsqueeze(dim=0), sampledImgs]))

images_for_show = torch.concatenate(images_for_show)
torchvision.utils.save_image(images_for_show, "./tmp/synthetic_positives.png", nrow=batch_size+1)
