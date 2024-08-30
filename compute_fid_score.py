import argparse
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torch.utils.data.dataset import Dataset
from torch.nn.functional import adaptive_avg_pool2d

class customized_dataset(Dataset):
    
    def __init__(self, path: str, transforms=None):
        self.data = np.load(path)
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, index: int):
        image = self.data[index]
        image = Image.fromarray(image)
        if self.transforms:
            image = self.transforms(image)
        return image

def get_activations(path, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- path       : dataset of numpy path
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    dataset = customized_dataset(path, transforms=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(path, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    act = get_activations(path, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_given_numpy(real_samples, fake_samples, batch_size, device, dims, num_workers=1):
    """Calculates the FID of numpy files."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(real_samples, model, batch_size, dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(fake_samples, model, batch_size, dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('--real_samples', default="./data_numpy/cifar10.npy", type=str, help='The numpy file for real samples')
    parser.add_argument('--fake_samples', default="./output/cifar10_tmp/generated_ddpm_0.01_000.npy", type=str, help='The numpy file for fake samples')
    args = parser.parse_args()
    # real_samples = "/afs/crc.nd.edu/user/d/dzeng2/data/cifar10/cifar10.npy"
    # fake_samples = "/afs/crc.nd.edu/user/d/dzeng2/code/clsp/output/cifar10_v14/generated_ddim_interpolation_0.01_000.npy"
    # fake_samples = "/afs/crc.nd.edu/user/d/dzeng2/code/clsp/output/cifar10_v20/generated_ddim_interpolation_0.1_000.npy"
    # fake_samples = "/afs/crc.nd.edu/user/d/dzeng2/data/cifar100/baseline_ddim_interpolation.npy"
    # fake_samples = "/afs/crc.nd.edu/user/d/dzeng2/code/clsp/output/cifar100_v3/generated_ddim_interpolation_0.1_000.npy"
    fid = calculate_fid_given_numpy(args.real_samples, args.fake_samples, batch_size=50, device='cuda:0', dims=2048)
    print(f"fid:{fid}")
