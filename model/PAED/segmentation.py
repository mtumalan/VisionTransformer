import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize

def compute_sdf(mask: np.ndarray):
    """
    Calcula mapas signed distance fields interno y externo a partir de una máscara binaria.

    Args:
        mask (np.ndarray): máscara binaria 2D (H, W), con valores {0,1}

    Returns:
        sdf_ext (np.ndarray): distancia normalizada desde el fondo a la frontera, 0 fuera del objeto
        sdf_int (np.ndarray): distancia normalizada desde el interior al borde, 0 fuera del objeto
    """

    # Validar máscara binaria
    mask = mask.astype(bool)

    # Distancia desde el fondo (pixels 0) al borde del objeto
    sdf_ext = distance_transform_edt(~mask).astype(np.float32)

    # Distancia desde el interior (pixels 1) al borde del objeto
    sdf_int = distance_transform_edt(mask).astype(np.float32)

    # Normalización a [0, 1]
    if sdf_ext.max() > 0:
        sdf_ext /= sdf_ext.max()

    if sdf_int.max() > 0:
        sdf_int /= sdf_int.max()

    return sdf_ext, sdf_int



def pixel_accuracy(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Precisión píxel a píxel.
    Args:
        gt: Ground truth tensor (B, H, W) o (B, 1, H, W), binario
        pred: Predicción tensor (B, H, W) o (B, 1, H, W), binario
    Returns:
        Precisión como tensor escalar.
    """
    gt = gt.squeeze().int()
    pred = pred.squeeze().int()
    correct = (gt == pred).float().sum()
    total = torch.numel(gt)
    return correct / total


def intersection_over_union(gt: torch.Tensor, pred: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Calcula el IoU para segmentación binaria.
    Args:
        gt: Ground truth tensor (B, H, W), binario
        pred: Predicción tensor (B, H, W), binario
        eps: Pequeña constante para evitar división por cero
    Returns:
        IoU como tensor escalar.
    """
    gt = gt.squeeze().int()
    pred = pred.squeeze().int()
    intersection = (gt & pred).float().sum()
    union = (gt | pred).float().sum()
    iou = (intersection + eps) / (union + eps)
    return iou


def dice_score(gt: torch.Tensor, pred: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Calcula el Dice score para segmentación binaria.
    Args:
        gt: Ground truth tensor (B, H, W), binario
        pred: Predicción tensor (B, H, W), binario
        eps: Pequeña constante para evitar división por cero
    Returns:
        Dice score como tensor escalar.
    """
    gt = gt.squeeze().int()
    pred = pred.squeeze().int()
    intersection = (gt & pred).float().sum()
    dice = (2 * intersection + eps) / (gt.float().sum() + pred.float().sum() + eps)
    return dice


class CrackSeg:
    @staticmethod
    def skeletonize(mask: torch.Tensor) -> torch.Tensor:
        """
        Convierte una máscara binaria en tensor (torch) a su esqueleto utilizando skimage.

        Args:
            mask (torch.Tensor): Tensor 2D (H, W) con valores en [0, 1] o binarios.

        Returns:
            torch.Tensor: Tensor 2D esquelético (binario).
        """
        # Convertimos a NumPy y aseguramos que sea binaria
        mask_np = mask.detach().cpu().numpy()
        mask_np_bin = (mask_np > 0.5).astype(np.uint8)

        # Aplicamos skeletonize de skimage
        skeleton_np = skimage_skeletonize(mask_np_bin)

        # Convertimos de vuelta a tensor
        skeleton_tensor = torch.from_numpy(skeleton_np).float().to(mask.device)

        return skeleton_tensor


