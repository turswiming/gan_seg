# ...existing code...
import torch
import torch.nn as nn

from torch_kdtree import build_kd_tree


class KDTreeDistanceLoss(nn.Module):
    """
    Compute nearest-neighbor distance from src -> tgt using torch_kdtree.
    Distances are clamped to `max_distance` before reduction.

    Args:
        max_distance (float): maximum distance to cap per-point distances.
        reduction (str): "mean", "sum" or "none" (per-batch item).
    Inputs:
        src: (B, N, 3) or (N, 3) tensor of query points
        tgt: (B, M, 3) or (M, 3) tensor of target points
    Returns:
        scalar tensor (reduction=="mean" or "sum") or tensor (B,) for "none"
    """
    def __init__(self, max_distance: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.max_distance = float(max_distance)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        # Cache for KDTree keyed by a point-cloud identifier (idx)
        self._cached_idx = None
        self._cached_tree = None
        self._cached_tgt_shape = None
        self._cached_tgt_device = None
        self._cached_tgt_dtype = None

    def forward(self, src: torch.Tensor, idx, tgt: torch.Tensor | None = None) -> torch.Tensor:
        """
        Query nearest-neighbour distances from `src` -> cached `tgt` KDTree.

        API:
          src: (N,3) float tensor (single point-cloud)
          idx: hashable id for the target point-cloud. If this matches the cached
               id, the cached KDTree is used. Otherwise `tgt` must be provided
               and a new KDTree will be built and cached under `idx`.
          tgt: (M,3) float tensor or None when cache hit.

        Returns:
          scalar (mean/sum) or tensor (per-item) depending on `reduction`.
        """
        # Validate src
        if not isinstance(src, torch.Tensor):
            raise TypeError("src must be a torch.Tensor of shape (N,3)")
        if src.dim() != 2 or src.shape[1] != 3:
            raise ValueError("src must have shape (N, 3)")

        device = src.device
        dtype = src.dtype

        # Cache control: if idx matches cached, we can ignore tgt
        tree = None
        if idx == self._cached_idx and self._cached_tree is not None:
            tree = self._cached_tree
        else:
            db = tgt.contiguous()
            # handle empty target
            if db.numel() == 0 or db.shape[0] == 0:
                tree = None
            else:
                tree = build_kd_tree(db)

            # update cache
            self._cached_idx = idx
            self._cached_tree = tree
            self._cached_tgt_shape = tuple(db.shape) if db is not None else None
            self._cached_tgt_device = db.device if db is not None else None
            self._cached_tgt_dtype = db.dtype if db is not None else None


        dists, _ = self._cached_tree.query(src, 1)

        dists = dists.squeeze(-1).to(device=device, dtype=dtype)

        # if distance is larger than max_distance, set to zero
        dists_clamped = torch.where(dists > self.max_distance, torch.zeros_like(dists), dists)        

        if self.reduction == "none":
            return dists_clamped
        elif self.reduction == "sum":
            return dists_clamped.sum()
        else:
            return dists_clamped.mean()
