import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Helpers                                                                 #
# --------------------------------------------------------------------- #

def _make_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Build a normalised 2-D Gaussian kernel.

    The kernel is constructed as the outer product of two 1-D Gaussians and
    normalised so that its elements sum to 1.

    Args:
        kernel_size (int): Side length of the square kernel.  Must be odd.
        sigma (float): Standard deviation of the Gaussian in pixels.
        device (torch.device): Target device for the returned tensor.

    Returns:
        Tensor: Kernel of shape ``(1, 1, kernel_size, kernel_size)``.

    Raises:
        ValueError: If ``kernel_size`` is even.
    """
    if kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd, got {kernel_size}.")

    half  = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g1d   = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g1d   = g1d / g1d.sum()                         # normalise to sum = 1
    g2d   = g1d.unsqueeze(1) @ g1d.unsqueeze(0)     # outer product -> (k, k)
    return g2d.unsqueeze(0).unsqueeze(0)             # (1, 1, k, k)


# --------------------------------------------------------------------- #
# Public API                                                              #
# --------------------------------------------------------------------- #

def gaussian_blur(
    input_tensor: torch.Tensor,
    kernel_size:  int   = 15,
    sigma:        float = 3.4,
) -> torch.Tensor:
    """Apply a separable Gaussian blur to every spectral band simultaneously.

    Uses a single depthwise ``F.conv2d`` call with ``groups=C`` instead of
    looping over bands, which is significantly faster on both CPU and GPU.
    Reflection padding is used to avoid dark borders at image edges.

    Args:
        input_tensor (Tensor): Hyperspectral image of shape ``(B, C, H, W)``.
        kernel_size (int): Side length of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian in pixels.

    Returns:
        Tensor: Blurred tensor of shape ``(B, C, H, W)``.
    """
    B, C, H, W = input_tensor.shape
    device = input_tensor.device

    # Build the 2-D kernel on the same device as the input (fixes CPU/CUDA mismatch)
    kernel = _make_gaussian_kernel(kernel_size, sigma, device)   # (1, 1, k, k)

    # Expand to (C, 1, k, k) so that groups=C applies one kernel per band
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)

    pad = kernel_size // 2

    # Reflect-pad the spatial dims to avoid zero-border artefacts
    x = F.pad(input_tensor, (pad, pad, pad, pad), mode='reflect')

    # Single depthwise convolution over all C bands — no Python loop required
    return F.conv2d(x, kernel, groups=C)


def downsample(input_tensor: torch.Tensor, scale_factor: int = 8) -> torch.Tensor:
    """Spatially downsample a tensor using bilinear interpolation.

    Args:
        input_tensor (Tensor): Image of shape ``(B, C, H, W)``.
        scale_factor (int): Integer downsampling factor (e.g. 8 -> H/8, W/8).

    Returns:
        Tensor: Downsampled image of shape ``(B, C, H//scale_factor, W//scale_factor)``.
    """
    B, C, H, W = input_tensor.shape
    out_h = H // scale_factor
    out_w = W // scale_factor

    # Pass target size explicitly to avoid the recompute_scale_factor deprecation
    # warning introduced in PyTorch >= 1.11
    return F.interpolate(
        input_tensor,
        size=(out_h, out_w),
        mode='bilinear',
        align_corners=False,
    )


def add_gaussian_noise(input_tensor: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """Add additive white Gaussian noise (AWGN) at a specified per-band SNR.

    The noise standard deviation is computed independently for each spectral
    band so that the target SNR is matched per band rather than globally.
    This is more accurate for hyperspectral data where band intensities vary
    substantially.

    Args:
        input_tensor (Tensor): Clean signal of shape ``(B, C, H, W)``.
        snr_db (float): Target signal-to-noise ratio in decibels.

    Returns:
        Tensor: Noisy tensor of the same shape as ``input_tensor``.
    """
    # Per-band signal power: mean over batch, height, width -> (1, C, 1, 1)
    signal_power = input_tensor.pow(2).mean(dim=(0, 2, 3), keepdim=True)

    snr_linear  = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear           # (1, C, 1, 1)
    noise_std   = noise_power.sqrt()                  # broadcast over B, H, W

    noise = noise_std * torch.randn_like(input_tensor)
    return input_tensor + noise


def apply_psf(
    input_tensor: torch.Tensor,
    kernel_size:  int   = 15,
    sigma:        float = 3.4,
    scale_factor: int   = 8,
    snr_db:       float = 20.0,
) -> torch.Tensor:
    """Simulate the full spatial PSF degradation pipeline.

    Applies three sequential steps that model how an HR-HSI is degraded to
    produce an observed LR-HSI:

    1. **Gaussian blur** — models the optical PSF / diffusion effect.
    2. **Bilinear downsampling** — models sensor spatial resolution limits.
    3. **AWGN** — models sensor read noise at the given SNR.

    Args:
        input_tensor (Tensor): Clean HR-HSI of shape ``(B, C, H, W)``.
        kernel_size (int): Gaussian blur kernel size (must be odd).
        sigma (float): Gaussian blur standard deviation in pixels.
        scale_factor (int): Spatial downsampling ratio.
        snr_db (float): Additive noise SNR in decibels.

    Returns:
        Tensor: Simulated LR-HSI of shape
            ``(B, C, H // scale_factor, W // scale_factor)``.
    """
    x = gaussian_blur(input_tensor, kernel_size=kernel_size, sigma=sigma)
    x = downsample(x, scale_factor=scale_factor)
    x = add_gaussian_noise(x, snr_db=snr_db)
    return x


# --------------------------------------------------------------------- #
# Quick sanity check                                                      #
# --------------------------------------------------------------------- #

if __name__ == '__main__':
    # Simulate a hyperspectral image: batch=1, 191 bands, 256x256 pixels
    dummy = torch.randn(1, 191, 256, 256)
    out   = apply_psf(dummy)

    expected_h = 256 // 8   # 32
    expected_w = 256 // 8   # 32
    assert out.shape == (1, 191, expected_h, expected_w), \
        f"Unexpected output shape: {out.shape}"

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print("apply_psf: OK")