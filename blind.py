import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun

from utils.toolkits import toolkits
from utils.torchkits import torchkits
from utils.blur_down import BlurDown
from data.data_info import DataInfo


# --------------------------------------------------------------------- #
# BlindNet — differentiable HSI degradation model                        #
# --------------------------------------------------------------------- #

class BlindNet(nn.Module):
    """Differentiable forward model that simulates HSI spatial-spectral degradation.

    Jointly learns a point-spread function (PSF) and a spectral response
    function (SRF) by minimising the consistency between the spectrally
    degraded HR-HSI and the spatially degraded HR-pci.

    The PSF is parameterised as a normalised ``(1, 1, K, K)`` convolution
    kernel; the SRF is parameterised as a ``(l, L, 1, 1)`` convolution
    kernel.  Both are projected onto the probability simplex after every
    gradient step via :meth:`Blind.check_weight`.

    Args:
        hs_bands (int): Number of hyperspectral bands L.
        ms_bands (int): Number of multispectral bands l.
        ker_size (int): PSF kernel side length K (must be odd).
        ratio (int): Spatial downsampling ratio between HR-pci and LR-HSI.

    Attributes:
        psf (nn.Parameter): Learnable PSF kernel, shape ``(1, 1, K, K)``.
        srf (nn.Parameter): Learnable SRF matrix, shape ``(l, L, 1, 1)``.
    """

    def __init__(self, hs_bands: int, ms_bands: int, ker_size: int, ratio: int) -> None:
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio    = ratio
        self.pad_num  = (ker_size - 1) // 2

        # Initialise PSF as a uniform kernel (sum = 1)
        psf = torch.full((1, 1, ker_size, ker_size), 1.0 / ker_size ** 2)
        self.psf = nn.Parameter(psf)

        # Initialise SRF as a uniform spectral average (each output band
        # is the mean of all input bands)
        srf = torch.full((ms_bands, hs_bands, 1, 1), 1.0 / hs_bands)
        self.srf = nn.Parameter(srf)

        self.blur_down = BlurDown()

    def forward(self, Y: torch.Tensor, Z: torch.Tensor):
        """Simulate spectral and spatial degradation.

        Args:
            Y (Tensor): HR-HSI of shape ``(1, L, H, W)``.
            Z (Tensor): HR-pci of shape ``(1, l, H, W)``.

        Returns:
            tuple[Tensor, Tensor]:
                - **Ylow** — spectrally degraded Y, shape ``(1, l, H, W)``.
                - **Zlow** — spatially degraded Z, shape ``(1, l, h, w)``.
        """
        # Spectral degradation: Y -> Ylow via learned SRF
        # check_weight already keeps SRF rows summing to 1, so no runtime
        # re-normalisation is needed here.
        Ylow = fun.conv2d(Y, self.srf, bias=None)           # (1, l, H, W)

        # Spatial degradation: Z -> Zlow via learned PSF + downsampling
        Zlow = self.blur_down(Z, self.psf, self.pad_num, self.ms_bands, self.ratio)

        return Ylow, Zlow


# --------------------------------------------------------------------- #
# Blind — PSF/SRF estimation trainer                                     #
# --------------------------------------------------------------------- #

class Blind(DataInfo):
    """Blind estimation of PSF and SRF from co-registered LR-HSI / HR-pci.

    Minimises the spectral-spatial consistency loss::

        L = ||SRF(Y) - BlurDown(PSF, Z)||_F

    where Y is the HR-HSI and Z is the HR-pci.  Both PSF and SRF are
    projected onto their respective simplices (non-negative, sum-to-one)
    after every gradient step.

    When ``blind=False`` the class skips estimation and uses the PSF/SRF
    already stored in the dataset (e.g. loaded from the ``.mat`` file).

    Args:
        ndata (int): Dataset index passed to :class:`DataInfo`.
        nratio (int): Spatial downsampling ratio.
        nsnr (int): Noise SNR index.
        kernel (int): PSF kernel half-width; actual kernel size is
            ``kernel + 1`` (must result in an odd side length).
        blind (bool): If ``False``, skip estimation and use ground-truth
            PSF/SRF from the dataset.
        lr (float): Learning rate for the Adam optimiser.

    Attributes:
        psf (ndarray or Tensor): Estimated (or ground-truth) PSF.
        srf (ndarray or Tensor): Estimated (or ground-truth) SRF.
        model (BlindNet): The degradation model (only when ``blind=True``).
    """

    def __init__(
        self,
        ndata:  int,
        nratio: int,
        nsnr:   int  = 0,
        kernel: int  = 1,
        blind:  bool = True,
        lr:     float = 1e-5,
    ) -> None:
        super().__init__(ndata, nratio, nsnr)
        self.strBR = 'BR.mat'
        self.blind = blind

        if not self.blind:
            print("Using ground-truth PSF and SRF from the dataset.")
            return

        print("Estimating PSF and SRF from observed data ...")

        self.lr       = lr
        self.ker_size = kernel + 1   # e.g. kernel=8 -> ker_size=9 (odd)

        if self.ker_size % 2 == 0:
            raise ValueError(
                f"ker_size must be odd (got ker_size={self.ker_size} from kernel={kernel}). "
                "Use an odd value of kernel (e.g. 8 -> ker_size=9).")

        # Zero-copy conversion: share memory with the underlying numpy arrays
        self.__hsi = torch.from_numpy(self.hsi)
        self.__pci = torch.from_numpy(self.pci)

        self.model     = BlindNet(
            self.hs_bands, self.ms_bands, self.ker_size, self.ratio).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        toolkits.check_dir(self.model_save_path)

    # ----------------------------------------------------------------- #

    def train(self, max_iter: int = 5000, verb: bool = True) -> None:
        """Run the blind PSF/SRF estimation loop.

        Args:
            max_iter (int): Number of gradient steps.
            verb (bool): If ``True``, print loss every 100 iterations.
        """
        if not self.blind:
            return

        hsi = self.__hsi.cuda()
        pci = self.__pci.cuda()

        for epoch in range(max_iter):
            # zero_grad before forward to avoid stale gradient accumulation
            self.optimizer.zero_grad()

            Ylow, Zlow = self.model(hsi, pci)
            loss       = torchkits.torch_norm(Ylow - Zlow)

            loss.backward()
            self.optimizer.step()

            # Project PSF and SRF onto the probability simplex
            self.model.apply(self.check_weight)

            if verb and (epoch + 1) % 100 == 0:
                print(f"epoch: {epoch + 1:5d}  lr: {self.lr}  loss: {loss.item():.6f}")

        torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')

        # Store estimated kernels as CPU tensors (no redundant numpy round-trip)
        self.psf = self.model.psf.detach().cpu().clone()
        self.srf = self.model.srf.detach().cpu().clone()

    # ----------------------------------------------------------------- #

    def get_save_result(self, is_save: bool = True) -> None:
        """Load the best checkpoint and optionally save PSF/SRF to disk.

        Args:
            is_save (bool): If ``True``, write ``BR.mat`` containing the
                estimated PSF (key ``'B'``) and SRF (key ``'R'``).
        """
        if not self.blind:
            return

        print("Saving estimated PSF and SRF ...")
        self.model.load_state_dict(
            torch.load(self.model_save_path + 'parameter.pkl'))

        # Squeeze batch/channel singleton dims: (1,1,K,K) -> (K,K), (l,L,1,1) -> (l,L)
        psf = self.model.psf.detach().cpu().numpy().squeeze()
        srf = self.model.srf.detach().cpu().numpy().squeeze()

        self.psf, self.srf = psf, srf

        if is_save:
            sio.savemat(self.save_path + self.strBR, {'B': psf, 'R': srf})

    # ----------------------------------------------------------------- #

    @staticmethod
    def check_weight(module: nn.Module) -> None:
        """Project PSF and SRF parameters onto their feasible sets.

        Called via ``model.apply()`` after every optimiser step.

        * **PSF**: clamp to ``[0, 1]``, then normalise so all elements sum to 1
          (projects onto the probability simplex over spatial positions).
        * **SRF**: clamp to ``[0, 10]``, then normalise each output-band row
          to sum to 1 (projects onto the row-stochastic simplex).

        Args:
            module (nn.Module): Any submodule of the network; the method is
                a no-op for modules that lack ``psf`` or ``srf`` attributes.
        """
        if hasattr(module, 'psf'):
            w = module.psf.data
            w.clamp_(0.0, 1.0)
            # Normalise: divide by total sum so PSF sums to 1
            w.div_(w.sum())

        if hasattr(module, 'srf'):
            w = module.srf.data
            w.clamp_(0.0, 10.0)
            # Normalise each output band (dim=1) so its row sums to 1
            row_sums = w.sum(dim=1, keepdim=True)
            w.div_(row_sums)