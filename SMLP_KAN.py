# --------------------------------------------------------
# Approximate or Perish: Spectral MLP-KAN Diffusion with Attentive Function
# Learning for Unsupervised Hyperspectral Image Restoration
#
# Hongcheng Jiang (1)*, Jingtang Ma (2)+, Gaoyuan Du (3),
# Jingchen Sun (4), Gengyuan Zhang (5), Zejun Zhang (6), Kai Luo (7)++
#
# (1) Liaoning Finance & Trade College
# (2) Amazon Web Services
# (3) University of Tennessee, Knoxville
# (4) University at Buffalo, SUNY
# (5) Ludwig Maximilian University of Munich, Germany
# (6) University of Southern California, Los Angeles
# (7) University of Virginia, Charlottesville
#
# Contact: hjq44@mail.umkc.edu  |  kl3pq@virgina.com
# --------------------------------------------------------
#
# This code is adapted from:
#   "A Spectral Diffusion Prior for Unsupervised Hyperspectral Image
#    Super-Resolution", IEEE TGRS, 2024
#   JianJun Liu — https://github.com/liuofficial/SDP
#
# We thank the original authors for their open-source implementation.
# --------------------------------------------------------
import time
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from data.data_info import DataInfo
from utils.torchkits import torchkits
from utils.toolkits import toolkits
from utils.blur_down import BlurDown
from utils.ema import EMA
from model.smlp_kan import MLPSkipNetConfig, Activation
from model.gaussian_diffusion import GaussianDiffusion
from blind import Blind
from metrics import psnr_loss, ssim, ergas, cc
from metrics import sam as sam_t


class SpecDiffusionNet(nn.Module):
    """SMLP-KAN denoising network wrapped with a Gaussian diffusion process.

    Learns a spectral diffusion prior over hyperspectral pixel spectra.
    The denoising backbone is an SMLP-KAN network (see model/smlp_kan.py)
    conditioned on sinusoidal timestep embeddings.

    Args:
        hs_bands (int): Number of hyperspectral bands (input/output dimension).
        layers (int): Number of layers in the SMLP-KAN backbone.
        timesteps (int): Total diffusion timesteps T.
    """

    def __init__(self, hs_bands: int, layers: int = 4, timesteps: int = 1000):
        super().__init__()
        self.hs_bands  = hs_bands
        self.timesteps = timesteps

        self.net = MLPSkipNetConfig(
            num_channels=hs_bands,
            skip_layers=tuple(range(1, layers)),
            num_hid_channels=512,
            num_layers=layers,
            num_time_emb_channels=64,
            activation=Activation.silu,
            use_norm=True,
            condition_bias=1.0,
            dropout=0.001,
            last_act=Activation.none,
            num_time_layers=2,
            time_last_act=False,
        ).make_model()

        self.gauss_diffusion = GaussianDiffusion(
            denoise_fn=self.net, timesteps=self.timesteps, improved=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the diffusion training loss for a batch of spectra.

        Args:
            X (Tensor): Spectral batch of shape ``(N, hs_bands)``.

        Returns:
            Tensor: Scalar diffusion loss.
        """
        return self.gauss_diffusion.train_losses(X)

    def sample(
        self,
        batch_size: int,
        device: str,
        continuous: bool = False,
        idx=None,
    ) -> torch.Tensor:
        """Sample spectra from the learned diffusion prior.

        Args:
            batch_size (int): Number of spectra to generate.
            device (str): Target device (e.g. ``'cuda'``).
            continuous (bool): If True, return the full denoising trajectory.
            idx: Optional timestep index for partial sampling.

        Returns:
            Tensor: Generated spectra of shape ``(batch_size, hs_bands)``.
        """
        shape = (batch_size, self.hs_bands)
        return self.gauss_diffusion.sample(
            shape=shape, device=device, continuous=continuous, idx=idx)



class SDM(DataInfo):
    """Trainer for the spectral diffusion prior (Stage 1).

    Fits a ``SpecDiffusionNet`` to the target hyperspectral image spectra
    using the standard DDPM objective.

    Args:
        ndata (int): Dataset index (0–3).
        nratio (int): Spatial downsampling ratio.
        nsnr (int): Noise SNR level index.
    """

    def __init__(self, ndata: int, nratio: int = 8, nsnr: int = 0):
        super().__init__(ndata, nratio, nsnr)

        lr          = [1e-2, 1e-2, 1e-2, 0.5e-2]
        self.lr     = lr[ndata]
        self.lr_fun = lambda epoch: 0.001 * max(1000 - epoch / 10, 1)
        layers      = [5, 5, 5, 5]

        self.model     = SpecDiffusionNet(self.hs_bands, layers=layers[ndata])
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fun)

        torchkits.get_param_num(self.model)
        toolkits.check_dir(self.model_save_path)
        print(self.model)
        self.estimate_flops()

        self.model_save_pkl  = self.model_save_path + 'spec.pkl'
        self.model_save_time = self.model_save_path + 't.mat'

    def estimate_flops(self) -> None:
        """Estimate FLOPs per denoising step using fvcore (optional)."""
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count_table

            class _Wrapper(nn.Module):
                """Thin wrapper that fixes the timestep for FLOPs counting."""
                def __init__(self, denoise_fn, timestep):
                    super().__init__()
                    self.denoise_fn = denoise_fn
                    self.timestep   = timestep

                def forward(self, x):
                    return self.denoise_fn(x, self.timestep)

            dummy_x       = torch.randn(1, self.hs_bands).cuda()
            dummy_t       = torch.zeros(1, dtype=torch.long).cuda()
            wrapped       = _Wrapper(self.model.net.cuda(), dummy_t)
            flops         = FlopCountAnalysis(wrapped, dummy_x)
            total_flops   = flops.total() * self.model.timesteps

            print(f"\nFLOPs per denoising step : {flops.total() / 1e6:.2f} MFLOPs")
            print(f"Total FLOPs (x{self.model.timesteps} steps)  : {total_flops / 1e9:.2f} GFLOPs")
            print(parameter_count_table(self.model.net))

        except ImportError:
            print("Install fvcore to enable FLOPs estimation.")
        except Exception as e:
            print("FLOPs estimation failed:", e)

    def convert_data(self, img: torch.Tensor) -> torch.Tensor:
        """Reshape a 4-D image tensor to a pixel-wise spectral matrix.

        Args:
            img (Tensor): Image of shape ``(1, B, H, W)``.

        Returns:
            Tensor: Pixel spectra of shape ``(H*W, B)``.
        """
        _, B, H, W = img.shape
        return img.reshape(B, H * W).permute(1, 0)

    def train(self, max_iter: int = 30000, batch_size: int = 1024) -> None:
        """Train the spectral diffusion prior.

        Args:
            max_iter (int): Number of gradient steps.
            batch_size (int): Mini-batch size (number of pixels per step).
        """
        cudnn.benchmark = True

        # torch.from_numpy shares memory with the numpy array (zero-copy),
        # whereas torch.tensor() would force a full data copy.
        fed_data   = self.convert_data(torch.from_numpy(self.tgt).float()).cuda()
        model      = self.model.cuda()
        model.train()
        ema        = EMA(model, 0.999)
        ema.register()
        time_start = time.perf_counter()

        for epoch in range(max_iter):
            lr  = self.optimizer.param_groups[0]['lr']
            idx = np.random.randint(0, fed_data.shape[0], size=(batch_size,))

            # zero_grad before forward to avoid accumulating stale gradients
            self.optimizer.zero_grad()
            loss = self.model(fed_data[idx, :])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            ema.update()

            if epoch % 1000 == 0:
                print(epoch, lr, torchkits.to_numpy(loss))
                torch.save(self.model.state_dict(), self.model_save_pkl)

            if epoch == max_iter - 1:
                ema.apply_shadow()
                torch.save(ema.model.state_dict(), self.model_save_pkl)

        sio.savemat(self.model_save_time, {'t': time.perf_counter() - time_start})
        torch.cuda.empty_cache()

    def show(self) -> None:
        """Visualise spectra sampled from the trained diffusion prior."""
        model = self.model.cuda()
        model.eval()
        model.load_state_dict(torch.load(self.model_save_pkl))
        spec = torchkits.to_numpy(model.sample(10, device='cuda', continuous=False))
        plt.figure(num=0)
        plt.plot(spec.T)
        plt.show()



class Target(nn.Module):
    """Learnable hyperspectral image represented as an ``nn.Parameter``.

    Args:
        hs_bands (int): Number of spectral bands.
        height (int): Spatial height.
        width (int): Spatial width.
    """

    def __init__(self, hs_bands: int, height: int, width: int):
        super().__init__()
        self.height = height
        self.width  = width
        self.img    = nn.Parameter(torch.ones(1, hs_bands, height, width))

    def get_image(self) -> torch.Tensor:
        """Return the current estimate of the HSI."""
        return self.img

    def check(self) -> None:
        """Project pixel values to the valid range [0, 1]."""
        self.img.data.clamp_(0.0, 1.0)



class SMLPKAN(DataInfo):
    """Full SMLP-KAN reconstruction network (Stage 2).

    Combines a spatial-spectral fidelity term (blur-downsampling + SRF
    degradation model) with the pre-trained spectral diffusion prior to
    recover the high-resolution HSI via score-based guidance.

    Args:
        ndata (int): Dataset index (0–3).
        nratio (int): Spatial downsampling ratio.
        nsnr (int): Noise SNR level index.
        psf (ndarray, optional): Point spread function kernel.
        srf (ndarray, optional): Spectral response function matrix.
    """

    def __init__(
        self,
        ndata:  int,
        nratio: int = 8,
        nsnr:   int = 0,
        psf=None,
        srf=None,
    ):
        super().__init__(ndata, nratio, nsnr)
        self.strX = 'X.mat'
        if psf is not None:
            self.psf = psf
        if srf is not None:
            self.srf = srf

        self.spec_net = SDM(ndata, nratio, nsnr)

        lrs                             = [1e-3, 1e-3, 2.5e-3, 8e-3]
        self.lr                         = lrs[ndata]
        self.ker_size                   = self.psf.shape[0]
        lams                            = [0.1, 0.1, 0.1, 1.0]
        self.lam_A, self.lam_B, self.lam_C = lams[ndata], 1, 1e-6
        self.lr_fun                     = lambda epoch: 1.0

        self.psf = torch.tensor(
            np.reshape(self.psf, (1, 1, self.ker_size, self.ker_size)))
        self.srf = torch.tensor(
            np.reshape(self.srf, (self.ms_bands, self.hs_bands, 1, 1)))

        # Zero-copy conversion from numpy arrays
        self.__hsi = torch.from_numpy(self.hsi).float()
        self.__pci = torch.from_numpy(self.pci).float()

        toolkits.check_dir(self.model_save_path)
        self.model_save_pkl = self.model_save_path + 'prior.pkl'
        self.blur_down      = BlurDown()

    def cpt_loss(
        self,
        X:   torch.Tensor,
        hsi: torch.Tensor,
        pci: torch.Tensor,
        psf: torch.Tensor,
        srf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the spatial-spectral fidelity loss.

        Enforces consistency with the observed LR-HSI (via blur-downsampling)
        and the HR-HSI (via spectral response convolution).

        Args:
            X (Tensor): Current HR-HSI estimate ``(1, B, H, W)``.
            hsi (Tensor): Observed DG-HSI.
            pci (Tensor): Observed HR-PCI.
            psf (Tensor): PSF kernel for spatial degradation.
            srf (Tensor): SRF kernel for spectral degradation.

        Returns:
            Tensor: Scalar fidelity loss ``lam_A * L_hsi + lam_B * L_pci``.
        """
        Y = self.blur_down(X, psf, int((self.ker_size - 1) / 2), self.hs_bands, self.ratio)
        Z = func.conv2d(X, srf, None)
        return (self.lam_A * func.mse_loss(Y, hsi, reduction='sum') +
                self.lam_B * func.mse_loss(Z, pci, reduction='sum'))

    def img_to_spec(self, X: torch.Tensor) -> torch.Tensor:
        """Flatten a 4-D HSI to a pixel-wise spectral matrix.

        Args:
            X (Tensor): Image of shape ``(1, B, H, W)``.

        Returns:
            Tensor: Pixel spectra of shape ``(H*W, B)``.
        """
        return X.reshape(self.hs_bands, -1).permute(1, 0)

    def train(self, gam: float = 1e-3) -> None:
        """Run the score-guided HR-HSI reconstruction loop.

        Iterates over the reverse diffusion trajectory, jointly minimising the
        spatial-spectral fidelity loss and the spectral prior score loss.

        Args:
            gam (float): Weight for the spectral prior loss term.
        """
        cudnn.benchmark = True
        model     = Target(self.hs_bands, self.height, self.width).cuda()
        opt       = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.lam_C)
        scheduler = optim.lr_scheduler.LambdaLR(opt, self.lr_fun)
        torchkits.get_param_num(model)

        hsi = self.__hsi.cuda()
        pci = self.__pci.cuda()
        psf = self.psf.cuda()
        srf = self.srf.cuda()

        # Load the pre-trained spectral prior and freeze its parameters
        self.spec_net.model.load_state_dict(
            torch.load(self.spec_net.model_save_pkl))
        self.spec_net.model.to(device=pci.device)
        self.spec_net.model.eval()
        for param in self.spec_net.model.parameters():
            param.requires_grad = False

        timesteps  = self.spec_net.model.timesteps
        model.train()
        ema        = EMA(model, 0.9)
        ema.register()
        time_start = time.perf_counter()

        for i in range(timesteps):
            lr = opt.param_groups[0]['lr']
            t  = torch.full(
                (self.height * self.width,), timesteps - 1 - i,
                device=pci.device, dtype=torch.long)

            # img and spec are constant within the inner loop (model parameters
            # do not change until opt.step()), so we compute them once here.
            img  = model.get_image()
            spec = self.img_to_spec(img)

            for _ in range(3):
                noise      = torch.randn_like(spec)
                xt         = self.spec_net.model.gauss_diffusion.q_sample(
                    spec, t, noise=noise)
                noise_pred = self.spec_net.model.gauss_diffusion.denoise_fn(xt, t)

                spat_spec_loss  = self.cpt_loss(img, hsi, pci, psf, srf)
                spec_prior_loss = func.mse_loss(noise_pred, noise, reduction='sum')
                loss            = spat_spec_loss + gam * spec_prior_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
                model.check()
                ema.update()

            scheduler.step()

            if i % 100 == 0 and self.ref is not None:
                img  = model.get_image()
                psnr = toolkits.psnr_fun(self.ref, torchkits.to_numpy(img))
                sam  = toolkits.sam_fun(self.ref, torchkits.to_numpy(img))
                print(i, psnr, sam, loss.data, lr)

        ema.apply_shadow()
        img        = torchkits.to_numpy(ema.model.get_image())
        img_tensor = torch.from_numpy(img)
        ref_tensor = torch.from_numpy(self.ref)

        SSIM   = ssim(img_tensor, ref_tensor, 11, 'mean', 1.)
        SAM    = sam_t(img_tensor, ref_tensor)
        EARGAS = ergas(img_tensor, ref_tensor)
        PSNR   = psnr_loss(img_tensor, ref_tensor, 1.)
        CC     = cc(img_tensor, ref_tensor)

        print(f"PSNR: {PSNR}  SSIM: {SSIM}  SAM: {SAM}  ERGAS: {EARGAS}  CC: {CC}")
        sio.savemat(
            self.save_path + self.strX,
            {'X': img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)})

        run_time = time.perf_counter() - time_start
        print(f"Training time: {run_time:.1f}s")
        torch.cuda.empty_cache()



if __name__ == '__main__':
    ndata, nratio, nsnr = 0, 8, 0

    # Stage 1: learn the spectral diffusion prior
    spec_net = SDM(ndata=ndata, nratio=nratio, nsnr=nsnr)
    spec_net.train()

    # Estimate PSF and SRF from the observed data
    blind = Blind(ndata=ndata, nratio=nratio, nsnr=nsnr, blind=True, kernel=8)
    blind.train()
    blind.get_save_result(is_save=True)

    # Stage 2: score-guided HR-HSI reconstruction
    gams = [1e-3, 1e-3, 1e-3, 1e-1]
    net  = SMLPKAN(ndata=ndata, nratio=nratio, nsnr=nsnr, psf=blind.psf, srf=blind.srf)
    net.train(gam=gams[ndata])