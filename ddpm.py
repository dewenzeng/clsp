
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.contiguous().view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):

    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value = 1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
    
    def forward(self, x, labels=None):
        t = torch.randint(self.T, size=(x.shape[0], ), device=x.device)
        noise = torch.randn_like(x)
        x_t = (extract(self.sqrt_alphas_bar, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise)
        if labels is None:
            # unconditional diffusion
            y = self.model(x_t, t)
        else:
            # conditional diffusion
            y = self.model(x_t, t, labels)
        # diffusion loss.
        loss_mse = F.mse_loss(y, noise, reduction='none')
        return loss_mse
    

class GaussianDiffusionSampler(nn.Module):

    def __init__(self, model, beta_1, beta_T, T, w=0.0):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_bar - 1))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # Equation 11 in "Denoising Diffusion Probabilistic Models".
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # var comes from Equation 7 in "Denoising Diffusion Probabilistic Models".
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var
    
    def p_mean_variance_with_anchor(self, x_t, t, x_anchor, weight):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model.forward_with_anchor(x_t, t, x_anchor, weight)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var
    
    def p_mean_variance_with_condition(self, x_t, t, labels):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var
    
    def get_xt(self, x_0, time):
        t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * time
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t
    
    def forward(self, x_T, labels=None, x_anchor=None, weight=None, sample_method="ddpm", ddim_sampling_timesteps=100, ddim_eta=1.0):
        """Diffusion sampling with pre-trained model.

        Args:
            x_T: The noise image
            labels: The conditional labels.
            x_anchor: The anchor image for interpolation
            weight: The interpolation weight
            sample_method: The sampling method. Choice from [ddpm, ddpm_interpolation, ddim, ddim_interpolation]
        Returns:
            The sampled image
        """
        if sample_method == "ddpm":
            return self.ddpm(x_T, labels)
        elif sample_method == "ddpm_interpolation":
            return self.ddpm_interpolation(x_T, x_anchor, weight)
        elif sample_method == "ddim":
            return self.ddim(x_T, labels, sampling_timesteps=100, eta=1.0)
        elif sample_method == "ddim_interpolation":
            return self.ddim_interpolation(x_T, x_anchor, weight, sampling_timesteps=ddim_sampling_timesteps, eta=ddim_eta)
        else:
            raise NotImplementedError(f"Sample method {sample_method} is not implemented.")

    def ddpm(self, x_T, labels=None):
        x_t = x_T
        for time_step in tqdm(reversed(range(0, self.T)), desc = 'sampling loop time step', total = self.T):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            if labels is None:
                mean, var= self.p_mean_variance(x_t=x_t, t=t)
            else:
                mean, var= self.p_mean_variance_with_condition(x_t=x_t, t=t, labels=labels)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def get_middle_features(self, x_0, time):
        t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * time
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        features = self.model.get_middle_features(x_t, t)
        return features
    
    def ddpm_interpolation(self, x_T, x_anchor, weight):
        x_t = x_T
        for time_step in tqdm(reversed(range(0, self.T)), desc = 'sampling loop time step', total = self.T):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            noise = torch.randn_like(x_T)
            xt_anchor = (
                extract(self.sqrt_alphas_bar, t, x_anchor.shape) * x_anchor +
                extract(self.sqrt_one_minus_alphas_bar, t, x_anchor.shape) * noise)
            mean, var= self.p_mean_variance_with_anchor(x_t=x_t, t=t, x_anchor=xt_anchor, weight=weight)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def ddim(self, x_T, labels=None, sampling_timesteps=100, eta=1.0):
        total_timesteps = self.T
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None
        x_t = x_T
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time
            if labels is None:
                pred_noise = self.model(x_t, t)
            else:
                pred_noise = self.model(x_t, t, labels)
                nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
                pred_noise = (1. + self.w) * pred_noise - self.w * nonEps
            x_start = (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise
            )

            if time_next < 0:
                x_t = x_start
                continue

            alpha = self.alphas_bar[time]
            alpha_next = self.alphas_bar[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)

            x_t = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return torch.clip(x_t, -1, 1)
    
    def ddim_interpolation(self, x_T, x_anchor, weight, sampling_timesteps=100, eta=1.0):
        total_timesteps = self.T
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None
        x_t = x_T
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time
            noise = torch.randn_like(x_T)
            xt_anchor = (
                extract(self.sqrt_alphas_bar, t, x_anchor.shape) * x_anchor +
                extract(self.sqrt_one_minus_alphas_bar, t, x_anchor.shape) * noise)
            pred_noise = self.model.forward_with_anchor(x_t, t, xt_anchor, weight)
            x_start = (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise
            )

            if time_next < 0:
                x_t = x_start
                continue

            alpha = self.alphas_bar[time]
            alpha_next = self.alphas_bar[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)

            x_t = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return torch.clip(x_t, -1, 1)
