import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from tqdm.auto import tqdm


# --------------------------------------------------------------------- #
# Beta schedules                                                          #
# --------------------------------------------------------------------- #

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Return a linearly spaced beta schedule scaled to ``timesteps``.

    Follows the schedule from Ho et al. (DDPM, NeurIPS 2020), Section 4.

    Args:
        timesteps (int): Total number of diffusion steps T.

    Returns:
        Tensor: Beta values of shape ``(T,)``, dtype float64.
    """
    scale      = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end   = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Return a cosine beta schedule as proposed by Nichol & Dhariwal (2021).

    Args:
        timesteps (int): Total number of diffusion steps T.
        s (float): Small offset to prevent beta from being too small near t=0.

    Returns:
        Tensor: Beta values of shape ``(T,)``, dtype float64, clipped to [0, 0.999].
    """
    steps           = timesteps + 1
    x               = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod  = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod  = alphas_cumprod / alphas_cumprod[0]
    betas           = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# --------------------------------------------------------------------- #
# Statistical helpers                                                     #
# --------------------------------------------------------------------- #

def normal_kl(
    mean1: torch.Tensor,
    logvar1: torch.Tensor,
    mean2: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL divergence between two diagonal Gaussian distributions.

    Both distributions are parameterised by their mean and log-variance.
    Returns the element-wise KL, i.e. the result has the same shape as the inputs.

    Args:
        mean1 (Tensor): Mean of the first distribution.
        logvar1 (Tensor): Log-variance of the first distribution.
        mean2 (Tensor): Mean of the second distribution.
        logvar2 (Tensor): Log-variance of the second distribution.

    Returns:
        Tensor: Element-wise KL divergence, same shape as inputs.
    """
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Fast approximation of the standard normal CDF Phi(x).

    Uses the tanh approximation from Hendrycks & Gimpel (2016).

    Args:
        x (Tensor): Input values.

    Returns:
        Tensor: Approximate CDF values in (0, 1), same shape as ``x``.
    """
    return 0.5 * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(
    x: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    """Log-likelihood of a discretised Gaussian evaluated at integer pixel values.

    Implements Eq. (13) from Ho et al. (DDPM, NeurIPS 2020).  The input is
    assumed to be normalised to ``[-1, 1]`` with 256 discrete levels.

    Args:
        x (Tensor): Observed values in ``[-1, 1]``.
        means (Tensor): Predicted Gaussian means, same shape as ``x``.
        log_scales (Tensor): Predicted log standard deviations, same shape as ``x``.

    Returns:
        Tensor: Log-probabilities, same shape as ``x``.
    """
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv    = torch.exp(-log_scales)

    plus_in  = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in  = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus          = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    log_cdf_delta         = torch.log((cdf_plus - cdf_min).clamp(min=1e-12))

    return torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, log_cdf_delta),
    )


# --------------------------------------------------------------------- #
# GaussianDiffusion                                                       #
# --------------------------------------------------------------------- #

class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process with DDPM and improved-DDPM support.

    Implements the forward process q(x_t | x_0) and the learned reverse
    process p_theta(x_{t-1} | x_t) for denoising diffusion probabilistic
    models (Ho et al., NeurIPS 2020; Nichol & Dhariwal, ICML 2021).

    All diffusion schedule tensors are registered as non-trainable buffers so
    that a single ``.to(device)`` call moves everything correctly and avoids
    repeated host-device synchronisation during sampling.

    Args:
        denoise_fn (nn.Module): The score network epsilon_theta(x_t, t).
            When ``improved=True`` it must output ``2 * C`` channels
            (noise prediction concatenated with variance logits).
        timesteps (int): Total diffusion steps T.
        beta_schedule (str): ``'linear'`` or ``'cosine'``.
        improved (bool): If True, use the improved-DDPM objective with
            learned variance (Nichol & Dhariwal, 2021).
    """

    def __init__(
        self,
        denoise_fn: nn.Module,
        timesteps:     int  = 1000,
        beta_schedule: str  = 'linear',
        improved:      bool = False,
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps  = timesteps
        self.improved   = improved

        # ── Beta schedule ─────────────────────────────────────────────────
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule '{beta_schedule}'. "
                             "Choose 'linear' or 'cosine'.")

        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = func.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # ── Register all schedule tensors as buffers ──────────────────────
        # Buffers are moved to the correct device automatically when
        # GaussianDiffusion.to(device) is called, eliminating the repeated
        # .to(t.device) inside _extract() that caused redundant host-device syncs.
        def buf(t: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(t.float(), requires_grad=False)

        self.register_buffer('betas',                        buf(betas))
        self.register_buffer('alphas_cumprod',               buf(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',          buf(alphas_cumprod_prev))

        # Forward process q(x_t | x_0) coefficients
        self.register_buffer('sqrt_alphas_cumprod',
                             buf(torch.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             buf(torch.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             buf(torch.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             buf(torch.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             buf(torch.sqrt(1.0 / alphas_cumprod - 1)))

        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # Clamp log-variance at t=0 where posterior_variance=0 (undefined log)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        )
        self.register_buffer('posterior_variance',
                             buf(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',
                             buf(posterior_log_variance_clipped))
        self.register_buffer('posterior_mean_coef1',
                             buf(betas * torch.sqrt(alphas_cumprod_prev)
                                 / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             buf((1.0 - alphas_cumprod_prev)
                                 * torch.sqrt(alphas)
                                 / (1.0 - alphas_cumprod)))

    # --------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # --------------------------------------------------------------------- #

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Gather schedule values at timestep indices t and reshape for broadcasting.

        Args:
            a (Tensor): 1-D schedule tensor of shape ``(T,)``.
            t (Tensor): Timestep indices of shape ``(B,)``.
            x_shape (tuple): Shape of the target tensor for broadcasting.

        Returns:
            Tensor: Gathered values of shape ``(B, 1, ..., 1)`` matching ``x_shape``.
        """
        out = a.gather(0, t).float()
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # --------------------------------------------------------------------- #
    # Forward process                                                         #
    # --------------------------------------------------------------------- #

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sample from the forward process q(x_t | x_0) in one step.

        Exploits the closed-form marginal:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_start (Tensor): Clean input x_0 of shape ``(B, ...)``.
            t (Tensor): Timestep indices of shape ``(B,)``.
            noise (Tensor, optional): Pre-sampled Gaussian noise.  If ``None``,
                noise is sampled from N(0, I).

        Returns:
            Tensor: Noisy sample x_t, same shape as ``x_start``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar   = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_m_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_m_alpha * noise

    def q_mean_variance(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """Return the mean, variance, and log-variance of q(x_t | x_0).

        Args:
            x_start (Tensor): Clean input x_0.
            t (Tensor): Timestep indices of shape ``(B,)``.

        Returns:
            tuple: ``(mean, variance, log_variance)`` each broadcastable to x_start.
        """
        mean         = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance     = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """Compute the posterior mean and variance of q(x_{t-1} | x_t, x_0).

        Args:
            x_start (Tensor): Predicted or true x_0.
            x_t (Tensor): Noisy sample at step t.
            t (Tensor): Timestep indices of shape ``(B,)``.

        Returns:
            tuple: ``(posterior_mean, posterior_variance, posterior_log_variance_clipped)``.
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance             = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # --------------------------------------------------------------------- #
    # Reverse process                                                         #
    # --------------------------------------------------------------------- #

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from x_t and predicted noise eps (inverse of q_sample).

        Args:
            x_t (Tensor): Noisy sample at step t.
            t (Tensor): Timestep indices of shape ``(B,)``.
            noise (Tensor): Predicted noise epsilon_theta(x_t, t).

        Returns:
            Tensor: Reconstructed x_0, same shape as ``x_t``.
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> tuple:
        """Compute the mean and variance of the learned reverse process p(x_{t-1} | x_t).

        Standard DDPM (``improved=False``): the model predicts noise only.
        Improved DDPM (``improved=True``): the model predicts noise and
        variance logits jointly; variance is interpolated between beta_t and
        beta_t_tilde as in Eq. (15) of Nichol & Dhariwal (2021).

        Args:
            x_t (Tensor): Noisy sample at step t of shape ``(B, C, ...)``.
            t (Tensor): Timestep indices of shape ``(B,)``.
            clip_denoised (bool): Whether to clamp the predicted x_0 to [-1, 1].

        Returns:
            tuple: ``(model_mean, model_variance, model_log_variance)``.
        """
        if not self.improved:
            pred_noise = self.denoise_fn(x_t, t)
            x_recon    = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0)
            model_mean, model_variance, model_log_variance = \
                self.q_posterior_mean_variance(x_recon, x_t, t)
        else:
            # Improved DDPM: model outputs [pred_noise | pred_variance_v]
            model_output        = self.denoise_fn(x_t, t)
            pred_noise, pred_v  = torch.chunk(model_output, 2, dim=1)

            # Interpolate log-variance between beta_t_tilde and beta_t (Eq. 15)
            min_log = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = self._extract(torch.log(self.betas), t, x_t.shape)
            frac    = (pred_v + 1.0) / 2.0          # map [-1,1] -> [0,1]
            model_log_variance = frac * max_log + (1.0 - frac) * min_log
            model_variance     = torch.exp(model_log_variance)

            x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0)
            model_mean, _, _ = self.q_posterior_mean_variance(x_recon, x_t, t)

        return model_mean, model_variance, model_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Sample x_{t-1} from the reverse process p(x_{t-1} | x_t).

        Args:
            x_t (Tensor): Noisy sample at step t.
            t (Tensor): Timestep indices of shape ``(B,)``.
            clip_denoised (bool): Whether to clamp predicted x_0 to [-1, 1].

        Returns:
            Tensor: Denoised sample x_{t-1}, same shape as ``x_t``.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # Mask out noise at the final step (t == 0) to avoid adding noise to x_0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        device: torch.device,
        continuous: bool = False,
        idx=None,
    ):
        """Run the full reverse diffusion chain from pure noise to a clean sample.

        Args:
            shape (tuple): Shape of the output tensor ``(B, C, ...)``.
            device (torch.device): Device for the initial noise tensor.
            continuous (bool): If True, return intermediate frames at timesteps
                selected by ``idx``.
            idx (array-like, optional): Boolean mask of length T; frame at step i
                is saved when ``idx[i] == 1``.  Required when ``continuous=True``.

        Returns:
            Tensor or list: Final sample ``(B, C, ...)`` when ``continuous=False``;
            list of CPU Tensors (one per selected frame) when ``continuous=True``.
        """
        batch_size = shape[0]
        img        = torch.randn(shape, device=device)

        timestep_iter = tqdm(
            reversed(range(self.timesteps)),
            desc='Sampling', total=self.timesteps)

        if not continuous:
            for i in timestep_iter:
                t   = torch.full((batch_size,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t)
            return img

        # Collect intermediate frames; defer .numpy() conversion until the end
        # to avoid stalling the GPU pipeline inside the hot loop.
        saved_tensors = [img.detach().cpu()]
        for i in timestep_iter:
            t   = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            if idx[i] == 1:
                saved_tensors.append(img.detach().cpu())
        return [t.numpy() for t in saved_tensors]

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: torch.device,
        continuous: bool = False,
        idx=None,
    ):
        """Generate new samples from the learned distribution.

        Thin wrapper around ``p_sample_loop``.

        Args:
            shape (tuple): Output shape ``(B, C, ...)``.
            device (torch.device): Target device.
            continuous (bool): Return the full denoising trajectory.
            idx: Frame-selection mask (see ``p_sample_loop``).

        Returns:
            Tensor or list: See ``p_sample_loop``.
        """
        return self.p_sample_loop(
            shape=shape, device=device, continuous=continuous, idx=idx)

    # --------------------------------------------------------------------- #
    # DDIM sampling (Song et al., ICLR 2021)                                 #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple,
        ddim_timesteps:    int   = 50,
        ddim_discr_method: str   = 'uniform',
        ddim_eta:          float = 0.0,
        clip_denoised:     bool  = True,
    ) -> torch.Tensor:
        """Accelerated sampling via DDIM (Song et al., ICLR 2021).

        Args:
            shape (tuple): Output shape ``(B, C, ...)``.  The device is inferred
                from ``self.denoise_fn``.
            ddim_timesteps (int): Number of DDIM denoising steps.
            ddim_discr_method (str): Timestep subsampling strategy;
                ``'uniform'`` or ``'quad'``.
            ddim_eta (float): Stochasticity parameter eta; 0 = deterministic DDIM,
                1 = equivalent to DDPM.
            clip_denoised (bool): Clamp predicted x_0 to [-1, 1].

        Returns:
            Tensor: Generated samples of shape ``shape``.
        """
        # Build the DDIM timestep subsequence
        if ddim_discr_method == 'uniform':
            c                = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.arange(0, self.timesteps, c)
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                np.linspace(0, np.sqrt(self.timesteps * 0.8), ddim_timesteps) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(
                f"Unknown DDIM discretization method '{ddim_discr_method}'.")

        ddim_timestep_seq      = ddim_timestep_seq + 1   # shift for correct alpha indexing
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.denoise_fn.parameters()).device
        img    = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(ddim_timesteps)), desc='DDIM sampling', total=ddim_timesteps):
            t      = torch.full((shape[0],), ddim_timestep_seq[i],      device=device, dtype=torch.long)
            prev_t = torch.full((shape[0],), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_t      = self._extract(self.alphas_cumprod, t,      img.shape)
            alpha_t_prev = self._extract(self.alphas_cumprod, prev_t, img.shape)

            pred_noise = self.denoise_fn(img, t)

            # Predicted x_0 (Eq. 12, Song et al. 2021)
            pred_x0 = (img - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Variance sigma_t (Eq. 16)
            sigma_t = ddim_eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))

            # Direction pointing to x_t (Eq. 12)
            pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise

            # x_{t-1} (Eq. 12)
            img = torch.sqrt(alpha_t_prev) * pred_x0 + pred_dir + sigma_t * torch.randn_like(img)

        return img

    # --------------------------------------------------------------------- #
    # Training objectives                                                     #
    # --------------------------------------------------------------------- #

    def train_losses(self, x_start: torch.Tensor) -> torch.Tensor:
        """Compute the standard DDPM MSE training loss (Ho et al., NeurIPS 2020).

        Randomly samples a timestep t for each example, applies the forward
        process, and returns the MSE between the true noise and the network
        prediction.

        Args:
            x_start (Tensor): Clean spectral input x_0 of shape ``(B, C)``.

        Returns:
            Tensor: Scalar MSE loss.
        """
        t     = torch.randint(0, self.timesteps, (x_start.shape[0],),
                              device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t   = self.q_sample(x_start, t, noise=noise)
        pred  = self.denoise_fn(x_t, t)
        return func.mse_loss(noise, pred)

    def train_ddpm_plus_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the improved-DDPM hybrid loss (Nichol & Dhariwal, ICML 2021).

        Combines a simple MSE noise-prediction term with a reweighted VLB term
        that trains the variance network.  The noise predictions are detached
        when computing the VLB so that only the variance head receives gradients
        from that term.

        Args:
            model (nn.Module): Denoising network (same as ``self.denoise_fn``).
            x_start (Tensor): Clean input x_0.
            t (Tensor): Timestep indices of shape ``(B,)``.

        Returns:
            Tensor: Scalar combined loss ``L_simple + L_vlb``.
        """
        noise   = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)

        model_output        = model(x_noisy, t)
        pred_noise, pred_v  = torch.chunk(model_output, 2, dim=1)

        # VLB term: train variance head only; detach noise predictions
        frozen_output = torch.cat([pred_noise.detach(), pred_v], dim=1)

        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_noisy, t)

        # Re-run p_mean_variance with frozen noise to get model mean/variance
        # Use a lightweight wrapper to avoid mutating self.denoise_fn
        _orig_fn = self.denoise_fn
        self.denoise_fn = lambda *a, **kw: frozen_output   # temporarily swap
        model_mean, _, model_log_var = self.p_mean_variance(x_noisy, t, clip_denoised=False)
        self.denoise_fn = _orig_fn                         # restore

        # KL loss for t > 0 (Eq. 14)
        kl  = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        kl  = kl.mean(dim=list(range(1, kl.ndim))) / math.log(2.0)

        # NLL loss for t = 0 (Eq. 13)
        nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_var)
        nll = nll.mean(dim=list(range(1, nll.ndim))) / math.log(2.0)

        # Reweight VLB by T/1000 as in the paper
        vlb_loss = torch.where(t == 0, nll, kl) * (self.timesteps / 1000.0)

        # Simple MSE loss on noise prediction
        mse_loss = (pred_noise - noise).pow(2).mean(dim=list(range(1, pred_noise.ndim)))

        return (mse_loss + vlb_loss).mean()

    # --------------------------------------------------------------------- #
    # Fast sampling (improved-DDPM spacing, Nichol & Dhariwal 2021)          #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def fast_sample(
        self,
        shape:              tuple,
        timestep_respacing: str  = '50',
        clip_denoised:      bool = True,
    ) -> np.ndarray:
        """Sample using the strided timestep spacing from Nichol & Dhariwal (2021).

        Constructs a non-uniform timestep subsequence that preserves the
        relative density of the original schedule and uses the improved-DDPM
        reverse step with learned variance.

        Args:
            shape (tuple): Output shape ``(B, C, ...)``.  Device is inferred
                from ``self.denoise_fn``.
            timestep_respacing (str): Comma-separated section counts, e.g.
                ``'50'`` or ``'10,10,10'``.
            clip_denoised (bool): Clamp predicted x_0 to [-1, 1].

        Returns:
            ndarray: Generated samples of shape ``shape``, on CPU.
        """
        # Build strided timestep sequence
        # (see https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/respace.py)
        section_counts = [int(x) for x in timestep_respacing.split(',')]
        size_per       = self.timesteps // len(section_counts)
        extra          = self.timesteps  % len(section_counts)
        timestep_seq   = []
        start_idx      = 0

        for i, count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < count:
                raise ValueError(
                    f"Cannot divide section of {size} steps into {count} sub-steps.")
            frac_stride = 1 if count <= 1 else (size - 1) / (count - 1)
            cur_idx     = 0.0
            for _ in range(count):
                timestep_seq.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            start_idx += size

        total_steps       = len(timestep_seq)
        timestep_prev_seq = np.append(np.array([-1]), timestep_seq[:-1])

        device = next(self.denoise_fn.parameters()).device
        img    = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(total_steps)), desc='Fast sampling', total=total_steps):
            t      = torch.full((shape[0],), timestep_seq[i],      device=device, dtype=torch.long)
            prev_t = torch.full((shape[0],), timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_t      = self._extract(self.alphas_cumprod,      t,          img.shape)
            alpha_t_prev = self._extract(self.alphas_cumprod_prev, prev_t + 1, img.shape)

            model_output       = self.denoise_fn(img, t)
            pred_noise, pred_v = torch.chunk(model_output, 2, dim=1)

            # Compute beta_t and beta_t_tilde for the strided step (Eq. 19)
            new_beta      = 1.0 - alpha_t / alpha_t_prev
            new_beta_tilde = new_beta * (1.0 - alpha_t_prev) / (1.0 - alpha_t)
            min_log = torch.log(new_beta_tilde)
            max_log = torch.log(new_beta)

            # Interpolate log-variance (Eq. 15)
            frac               = (pred_v + 1.0) / 2.0
            model_log_variance = frac * max_log + (1.0 - frac) * min_log

            # Reconstruct x_0
            x_recon = self.predict_start_from_noise(img, t, pred_noise)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0)

            # Posterior mean
            coef1      = new_beta * torch.sqrt(alpha_t_prev) / (1.0 - alpha_t)
            coef2      = (1.0 - alpha_t_prev) * torch.sqrt(1.0 - new_beta) / (1.0 - alpha_t)
            model_mean = coef1 * x_recon + coef2 * img

            noise        = torch.randn_like(img)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            img          = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return img.cpu().numpy()