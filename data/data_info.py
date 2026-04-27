import os
from typing import List
import numpy as np
import scipy.io as sio

from utils.toolkits import toolkits


class DataInfo:
    """Base class for loading and preprocessing hyperspectral fusion datasets.

    Reads a ``.mat`` file that contains a co-registered LR-HSI / HR-pci pair
    and exposes all tensors as channel-first (NCHW) float32 numpy arrays.
    Subclasses inherit these arrays and the derived metadata attributes to
    avoid duplicating I/O and preprocessing logic.

    Expected ``.mat`` keys
    ----------------------
    LRMS : ndarray, shape (h, w, L)
        Low-resolution hyperspectral image (LR-HSI).
    PAN  : ndarray, shape (H, W)  or  (H, W, l)
        High-resolution multispectral / panchromatic image (HR-pci).
    HRMS : ndarray, shape (H, W, L)
        High-resolution hyperspectral reference (ground truth).
    K    : ndarray, shape (k, k), optional
        Point-spread-function (PSF) kernel.  Falls back to a uniform kernel
        whose size is inferred from the spatial upsampling ratio when absent.
    R    : ndarray, shape (l, L), optional
        Spectral response function (SRF) matrix.  Falls back to an all-ones
        matrix when absent.

    Directory layout
    ----------------
    ``<gen_path>/<folder_name>/<data_name><ratio><noise>.mat``

    For example::

        data/PaviaC_256_8_4.mat

    Args:
        ndata (int): Dataset index that selects from ``folder_names``,
            ``data_names``, etc.  Must satisfy ``0 <= ndata < len(data_names)``.
        nratio (int): Spatial downsampling ratio between HR-pci and LR-HSI.
        nsnr (int): Noise-level index that selects from ``noise``.

    Attributes:
        hsi (ndarray): LR-HSI, shape ``(1, L, h, w)``, float32.
        pci (ndarray): HR-pci, shape ``(1, l, H, W)``, float32.
        ref (ndarray): HR-HSI reference, shape ``(1, L, H, W)``, float32.
        tgt (ndarray): Diffusion training target (same as LR-HSI by default),
            shape ``(1, L, h, w)``, float32.
        psf (ndarray): PSF kernel, shape ``(k, k)``, float32.
        srf (ndarray): SRF matrix, shape ``(l, L)``, float32.
        hs_bands (int): Number of hyperspectral bands L.
        ms_bands (int): Number of multispectral bands l.
        ratio (int): Spatial upsampling ratio H / h.
        height (int): HR spatial height H.
        width (int): HR spatial width W.
        file_path (str): Absolute path to the loaded ``.mat`` file.
        save_path (str): Output directory for results.
        model_save_path (str): Sub-directory for saved model checkpoints.
    """

    # --------------------------------------------------------------------- #
    # Dataset registry — extend these lists to add new datasets              #
    # --------------------------------------------------------------------- #
    gen_path     : str       = ''           # root directory; override per machine
    folder_names : List[str] = ['data/']
    data_names   : List[str] = ['PaviaC_256_']
    noise        : List[str] = ['_4']

    def __init__(self, ndata: int = 0, nratio: int = 8, nsnr: int = 0) -> None:
        # Log which subclass is being initialised
        print(f"{self.__class__.__name__} is running")

        # ── Input validation ─────────────────────────────────────────────
        n_datasets = len(self.data_names)
        if not (0 <= ndata < n_datasets):
            raise ValueError(
                f"ndata={ndata} is out of range; "
                f"expected 0 <= ndata < {n_datasets}.")

        # ── Build file paths ──────────────────────────────────────────────
        self.file_path = os.path.join(
            self.gen_path,
            self.folder_names[ndata],
            f"{self.data_names[ndata]}{nratio}{self.noise[nsnr]}.mat",
        )
        print(f"Loading: {self.file_path}")

        # ── Load .mat file ────────────────────────────────────────────────
        mat = sio.loadmat(self.file_path)

        # LR-HSI  (h, w, L)  and  HR-pci  (H, W) or (H, W, l)
        hsi: np.ndarray = mat['LRMS']
        pci: np.ndarray = mat['PAN']

        # Ensure pci has an explicit band axis: (H, W) -> (H, W, 1)
        if pci.ndim == 2:
            pci = np.expand_dims(pci, axis=-1)

        ref: np.ndarray = mat['HRMS']   # HR reference  (H, W, L)
        tgt: np.ndarray = mat['LRMS']   # Training target = LR-HSI

        # ── PSF and SRF ───────────────────────────────────────────────────
        if 'K' in mat:
            psf: np.ndarray = mat['K']  # (k, k)
            srf: np.ndarray = mat['R']  # (l, L)
        else:
            # Fall back to uniform kernels sized from the spatial ratio
            ratio_h = pci.shape[0] // hsi.shape[0]
            ratio_w = pci.shape[1] // hsi.shape[1]
            psf = np.ones((ratio_h, ratio_w), dtype=np.float32)
            srf = np.ones((pci.shape[-1], hsi.shape[-1]), dtype=np.float32)

        # ── Cast to float32 ───────────────────────────────────────────────
        hsi = hsi.astype(np.float32)
        pci = pci.astype(np.float32)
        ref = ref.astype(np.float32)
        tgt = tgt.astype(np.float32)
        self.psf = psf.astype(np.float32)
        self.srf = srf.astype(np.float32)

        # ── Convert to channel-first layout (NCHW) ────────────────────────
        self.hsi = toolkits.channel_first(hsi)   # (1, L, h, w)
        self.pci = toolkits.channel_first(pci)   # (1, l, H, W)
        self.ref = toolkits.channel_first(ref)   # (1, L, H, W)
        self.tgt = toolkits.channel_first(tgt)   # (1, L, h, w)

        # ── Derived metadata ──────────────────────────────────────────────
        self.hs_bands: int = self.hsi.shape[1]   # L
        self.ms_bands: int = self.pci.shape[1]   # l
        self.ratio:    int = int(self.pci.shape[-1] / self.hsi.shape[-1])
        self.height:   int = self.pci.shape[2]   # H
        self.width:    int = self.pci.shape[3]   # W

        # ── Output paths ──────────────────────────────────────────────────
        self.save_path = os.path.join(
            self.gen_path,
            self.folder_names[ndata],
            self.__class__.__name__,
            f"t1000{self.data_names}{self.noise[nsnr]}",
            "",   # trailing separator so subclasses can append filenames directly
        )
        self.model_save_path = os.path.join(self.save_path, 'model', '')