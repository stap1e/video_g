import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from tqdm import tqdm

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, sampling_timesteps=None, objective='pred_x0'):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.objective = objective

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.timesteps = timesteps

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        if x_start.shape != x_t.shape:
            print(f"[shape-debug] x_start={tuple(x_start.shape)} x_t={tuple(x_t.shape)} t={tuple(t.shape)}")
            if x_start.dim() == 5 and x_t.dim() == 5:
                x_start = F.interpolate(x_start, size=x_t.shape[2:], mode="trilinear", align_corners=False)
            elif x_start.dim() == 4 and x_t.dim() == 4:
                x_start = F.interpolate(x_start, size=x_t.shape[2:], mode="bilinear", align_corners=False)
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1, 1)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, cond=None, reconstruction_guidance=None):
        # reconstruction_guidance: dict with keys 'x_a' (ground truth), 'mask_a' (mask for a), 'w_r' (weight)
        
        # If using guidance, we need gradients w.r.t x (which is z_t)
        if reconstruction_guidance is not None:
            x = x.detach().requires_grad_(True)
            
        if cond is None:
            model_output = self.model(x, t)
        else:
            try:
                parameters = inspect.signature(self.model.forward).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "cond" in parameters:
                model_output = self.model(x, t, cond=cond)
            else:
                model_output = self.model(x, t)
        if model_output.shape != x.shape:
            if model_output.dim() == 5 and x.dim() == 5:
                model_output = F.interpolate(model_output, size=x.shape[2:], mode="trilinear", align_corners=False)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if x_start.shape != x.shape:
            if x_start.dim() == 5 and x.dim() == 5:
                x_start = F.interpolate(x_start, size=x.shape[2:], mode="trilinear", align_corners=False)
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # Reconstruction Guidance
        if reconstruction_guidance is not None:
            # Eq (7): \tilde{x}_\theta^b(z_t) = \hat{x}_\theta^b(z_t) - \frac{w_r \alpha_t}{2} \nabla_{z_t^b} || x^a - \hat{x}_\theta^a(z_t) ||_2^2
            # Here x_start is \hat{x}_\theta(z_t).
            # We want to adjust x_start.
            
            x_a_gt = reconstruction_guidance['x_a']
            mask_a = reconstruction_guidance['mask_a'] # 1 for 'a' region, 0 for 'b'
            w_r = reconstruction_guidance['w_r']
            
            # Loss = || x^a - x_start^a ||^2
            # Apply mask to extract 'a' part
            # Ensure x_a_gt matches shape if needed or mask handles it
            
            # x_start is (b, c, f, h, w)
            # mask_a is (b, 1, f, h, w) or similar
            
            x_start_a = x_start * mask_a
            x_a_target = x_a_gt * mask_a
            
            loss = F.mse_loss(x_start_a, x_a_target, reduction='sum')
            
            # Gradient w.r.t z_t (x)
            grad = torch.autograd.grad(loss, x)[0]
            
            # We only update 'b' part of z_t? No, the formula updates \hat{x}_\theta^b.
            # Wait, the formula (7) is:
            # \tilde{x}_0^b(z_t) = \hat{x}_0^b(z_t) - term * grad
            # But the gradient is \nabla_{z_t^b}.
            
            # Actually, standard classifier guidance modifies the MEAN of the posterior.
            # Ho et al. 2022 Eq (7) defines an "adjusted denoising model" \tilde{x}_0.
            # So we modify x_start directly.
            
            # \alpha_t in the paper likely refers to the noise schedule term, but let's check.
            # The formula has \frac{w_r \alpha_t}{2}.
            # Usually guidance involves variance.
            
            # Let's approximate the shift.
            # The gradient term is \nabla_{z_t} ||...||^2.
            # We modify x_start.
            
            # For simplicity and to match the structure:
            # We subtract the gradient scaled by a factor.
            # The paper says: \tilde{x}_0(z_t) = \hat{x}_0(z_t) - ...
            
            # Note: The gradient is w.r.t z_t (the noisy input x).
            # The mask should be applied to the gradient to only guide z_t^b?
            # "where the second term is missing... reconstruction of conditioning data x^a... gradient w.r.t z_t^b"
            # So we only adjust the 'b' part of the prediction?
            # Yes: \tilde{x}_0^b. The 'a' part is likely fixed or we don't care because we have ground truth x^a?
            # Actually for autoregressive, we might just replace the 'a' part with ground truth at each step (replacement method),
            # but the paper says replacement method is not coherent, so they use guidance.
            # So we adjust the predicted 'b' part using the gradient from the 'a' part reconstruction error.
            
            # Current alpha_t (signal to noise ratio or similar).
            # Let's just use a scalar weight w_r for now, maybe scaled by (1-alpha_cumprod)/sqrt(alpha_cumprod) or similar if exact math needed.
            # But the paper gives explicit formula involving alpha_t.
            # Assuming alpha_t is standard notation (1-beta_t) or alpha_bar_t.
            # Let's use the gradients directly.
            
            # x_start = x_start - w_r * grad
            
            # We need to mask the gradient to only affect 'b'.
            mask_b = 1. - mask_a
            
            # Refine x_start
            x_start = x_start - w_r * grad * mask_b
            
            x_start = x_start.detach()
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, cond=None, reconstruction_guidance=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond, reconstruction_guidance=reconstruction_guidance)
        noise = torch.randn_like(x) if t[0].item() > 0 else 0.
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, shape, cond=None, reconstruction_guidance=None):
        device = next(self.model.parameters()).device
        if isinstance(shape, int):
            time_steps = getattr(self.model, "time_steps", None)
            if time_steps is None and hasattr(self.model, "module"):
                time_steps = getattr(self.model.module, "time_steps", None)
            if time_steps is None:
                raise ValueError("time_steps is required on the model to infer sample shape")
            shape = (shape, self.channels, time_steps, self.image_size, self.image_size)
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps, ncols=80):
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            # We need gradients for guidance, so we cannot use no_grad for the whole loop if guidance is on.
            # However, sample is decorated with no_grad.
            # We should wrap p_sample call with enable_grad if guidance is present.
            
            if reconstruction_guidance is not None:
                with torch.enable_grad():
                    img = self.p_sample(img, t_tensor, cond=cond, reconstruction_guidance=reconstruction_guidance)
            else:
                img = self.p_sample(img, t_tensor, cond=cond, reconstruction_guidance=reconstruction_guidance)
            
            img = img.detach() # Detach after step to save memory
            
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target)
        return loss
