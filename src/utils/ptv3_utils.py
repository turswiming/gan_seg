"""
Utilities for loading PointTransformerV3 pretrained models.
"""

import os
import torch
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")


def download_file(url, filepath, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def load_ptv3_pretrained(model, pretrained_path=None, pretrained_name=None, strict=False):
    """
    Load pretrained PointTransformerV3 weights.
    
    Args:
        model: PTV3MaskPredictor or PTV3PanopticPredictor model
        pretrained_path: Local path to pretrained weights (.pth or .pt file)
        pretrained_name: Name of pretrained model
            Options: 
            - 'sonata' - Sonata self-supervised pretrained model (recommended)
            - 'sonata_small' - Sonata small self-supervised pretrained model
            - 'scannet-semseg-pt-v3m1-0-base' - ScanNet supervised model
            - 'scannet-semseg-pt-v3m1-1-ppt-extreme' - ScanNet + PPT model
            - 'nuscenes-semseg-pt-v3m1-0-base' - nuScenes supervised model
            - 'waymo-semseg-pt-v3m1-0-base' - Waymo supervised model
        strict: Whether to strictly enforce that the keys match
    
    Returns:
        model: Model with loaded weights
    """
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found at {pretrained_path}")
        # Check file size (should be at least a few MB for a valid checkpoint)
        file_size = os.path.getsize(pretrained_path)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            raise ValueError(f"Checkpoint file {pretrained_path} is too small ({file_size} bytes). File may be corrupted.")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {pretrained_path}. "
                f"The file may be corrupted or in an unsupported format. Error: {e}"
            ) from e
    elif pretrained_name:
        # Handle Sonata models (sonata, sonata_small, etc.)
        if pretrained_name.lower().startswith('sonata'):
            # Extract model name (e.g., "sonata_small" -> "sonata_small", "sonata" -> "sonata")
            sonata_name = pretrained_name.lower()
            checkpoint = download_sonata_weights(name=sonata_name)
        else:
            checkpoint = download_huggingface_weights(pretrained_name)
    else:
        raise ValueError("Either pretrained_path or pretrained_name must be provided")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'backbone' in checkpoint:
            # If checkpoint contains backbone weights
            state_dict = checkpoint['backbone']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Filter state dict to match model structure
    model_state_dict = model.state_dict()
    
    # Try to load backbone weights
    backbone_state_dict = {}
    mask_head_state_dict = {}
    
    for key, value in state_dict.items():
        # Remove 'module.' prefix if present
        key_clean = key.replace('module.', '')
        
        # Check if it's a backbone weight
        if key_clean.startswith('backbone.') or key_clean.startswith('enc.') or key_clean.startswith('dec.'):
            backbone_key = key_clean.replace('backbone.', '')
            if backbone_key in model.backbone.state_dict():
                backbone_state_dict[backbone_key] = value
        elif key_clean.startswith('mask_head.') or key_clean.startswith('semantic_head.'):
            mask_head_key = key_clean.replace('mask_head.', '').replace('semantic_head.', '')
            if hasattr(model, 'mask_head') and mask_head_key in model.mask_head.state_dict():
                mask_head_state_dict[mask_head_key] = value
    
    # If no backbone prefix found, try direct matching
    if not backbone_state_dict:
        for key, value in state_dict.items():
            key_clean = key.replace('module.', '')
            # Try to match with backbone structure
            if any(k in key_clean for k in ['embedding', 'enc', 'dec', 'norm', 'attn', 'mlp']):
                # Try to load into backbone
                if key_clean in model.backbone.state_dict():
                    backbone_state_dict[key_clean] = value
    
    # Load backbone weights
    if backbone_state_dict:
        missing_keys, unexpected_keys = model.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        print(f"Loaded {len(backbone_state_dict)} backbone weights")
        
        # Filter out BatchNorm running stats (running_mean, running_var) as they are runtime statistics
        # These are normal to be missing and will be recomputed during training
        important_missing_keys = [
            k for k in missing_keys 
            if not (k.endswith('.running_mean') or k.endswith('.running_var') or 
                   k.endswith('.num_batches_tracked'))
        ]
        
        if important_missing_keys:
            print(f"Missing important backbone keys: {len(important_missing_keys)}")
            if len(important_missing_keys) <= 20:
                print("Missing keys:", important_missing_keys)
            else:
                print("First 20 missing keys:", important_missing_keys[:20])
                print(f"... and {len(important_missing_keys) - 20} more")
        else:
            print(f"All important weights loaded. ({len(missing_keys)} BatchNorm runtime stats will be recomputed)")
        
        if unexpected_keys:
            print(f"Unexpected backbone keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                print("Unexpected keys:", unexpected_keys)
    else:
        # Try loading entire state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            print(f"Loaded model weights (strict={strict})")
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Warning: Could not load weights directly: {e}")
            print("Model will use random initialization")
    
    return model


def download_sonata_weights(cache_dir=None, name="sonata", repo_id="facebook/sonata"):
    """
    Download Sonata self-supervised pretrained weights from HuggingFace Hub.
    Sonata is available at: https://github.com/facebookresearch/sonata
    
    Args:
        cache_dir: Directory to cache downloaded weights (default: ~/.cache/sonata/ckpt)
        name: Model name (default: "sonata")
        repo_id: HuggingFace repository ID (default: "facebook/sonata")
    
    Returns:
        checkpoint: Loaded checkpoint dict
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/sonata/ckpt")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use HuggingFace Hub if available
    if HF_HUB_AVAILABLE:
        try:
            print(f"Loading Sonata checkpoint from HuggingFace: {name} ...")
            # hf_hub_download returns the local path to the downloaded file
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{name}.pth",
                revision="main",
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"Failed to download from HuggingFace Hub: {e}")
            print("Falling back to direct download...")
            # Fallback to direct download
            checkpoint_path = _download_sonata_fallback(cache_dir, name)
    else:
        print("Warning: huggingface_hub not available. Using fallback download method.")
        print("Install with: pip install huggingface_hub")
        checkpoint_path = _download_sonata_fallback(cache_dir, name)
    
    # Check file size and integrity
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Sonata checkpoint not found at {checkpoint_path}")
    
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
        print(f"Warning: Checkpoint file is suspiciously small ({file_size} bytes). Attempting to re-download...")
        try:
            os.remove(checkpoint_path)
            if HF_HUB_AVAILABLE:
                checkpoint_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{name}.pth",
                    revision="main",
                    cache_dir=cache_dir,
                    force_download=True,
                )
            else:
                checkpoint_path = _download_sonata_fallback(cache_dir, name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to re-download Sonata weights: {e}. "
                "Please download manually from https://huggingface.co/facebook/sonata"
            ) from e
    
    print(f"Loading Sonata checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # If loading fails, try removing the corrupted file and re-downloading
        if "MARK" in str(e) or "corrupted" in str(e).lower():
            print(f"Checkpoint file appears corrupted (error: {e}). Attempting to re-download...")
            try:
                os.remove(checkpoint_path)
                if HF_HUB_AVAILABLE:
                    checkpoint_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{name}.pth",
                        revision="main",
                        cache_dir=cache_dir,
                        force_download=True,
                    )
                else:
                    checkpoint_path = _download_sonata_fallback(cache_dir, name)
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as re_download_error:
                raise RuntimeError(
                    f"Failed to load checkpoint and re-download failed: {re_download_error}. "
                    "Please manually download from https://huggingface.co/facebook/sonata"
                ) from re_download_error
        else:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}: {e}"
            ) from e
    return checkpoint


def _download_sonata_fallback(cache_dir, name="sonata"):
    """
    Fallback method to download Sonata weights using direct URLs.
    Used when HuggingFace Hub is not available.
    
    Args:
        cache_dir: Directory to cache downloaded weights
        name: Model name (default: "sonata")
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sonata repository URLs (fallback)
    # Note: Fallback URLs may only support "sonata", not "sonata_small"
    sonata_urls = [
        f"https://github.com/facebookresearch/sonata/releases/download/v1.0/{name}.pth",
        f"https://dl.fbaipublicfiles.com/sonata/{name}.pth",
    ]
    
    checkpoint_path = os.path.join(cache_dir, f"{name}.pth")
    
    if not os.path.exists(checkpoint_path):
        print("Downloading Sonata self-supervised pretrained model...")
        print("Note: If download fails, please download manually from:")
        print("  https://huggingface.co/facebook/sonata")
        
        downloaded = False
        for url in sonata_urls:
            try:
                download_file(url, checkpoint_path)
                if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
                    downloaded = True
                    break
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                continue
        
        if not downloaded:
            raise FileNotFoundError(
                "Could not download Sonata weights automatically. "
                "Please download manually from https://huggingface.co/facebook/sonata "
                "and place the checkpoint file at the expected location."
            )
    
    return checkpoint_path


def download_huggingface_weights(model_name, cache_dir=None):
    """
    Download pretrained weights from HuggingFace.
    
    Args:
        model_name: Name of the model
        cache_dir: Directory to cache downloaded weights
    
    Returns:
        checkpoint: Loaded checkpoint dict
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/ptv3_weights")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # HuggingFace model repository
    base_url = "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main"
    
    # Try different possible file names
    possible_files = [
        f"{model_name}/model.pth",
        f"{model_name}/checkpoint.pth",
        f"{model_name}/best_model.pth",
        f"{model_name}/model.pt",
        f"{model_name}/checkpoint.pt",
        f"{model_name}/best_model.pt",
    ]
    
    checkpoint_path = None
    for file_path in possible_files:
        local_path = os.path.join(cache_dir, file_path.replace('/', '_'))
        url = f"{base_url}/{file_path}"
        
        try:
            if not os.path.exists(local_path):
                print(f"Downloading {file_path} from HuggingFace...")
                download_file(url, local_path)
            
            if os.path.exists(local_path):
                checkpoint_path = local_path
                break
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")
            continue
    
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not download pretrained weights for {model_name}. "
            f"Please download manually from https://huggingface.co/Pointcept/PointTransformerV3"
        )
    
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not download pretrained weights for {model_name}. "
            f"Please download manually from https://huggingface.co/Pointcept/PointTransformerV3"
        )
    
    # Check file integrity
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
        raise ValueError(f"Checkpoint file {checkpoint_path} is too small ({file_size} bytes). File may be corrupted.")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from {checkpoint_path}. "
            f"The file may be corrupted or in an unsupported format. Error: {e}"
        ) from e
    return checkpoint


def load_ptv3_backbone_only(model, pretrained_path=None, pretrained_name=None):
    """
    Load only the backbone weights, useful when adapting pretrained semantic segmentation
    models for instance segmentation.
    
    Args:
        model: PTV3MaskPredictor or PTV3PanopticPredictor model
        pretrained_path: Local path to pretrained weights
        pretrained_name: Name of pretrained model from HuggingFace
    
    Returns:
        model: Model with loaded backbone weights
    """
    return load_ptv3_pretrained(model, pretrained_path, pretrained_name, strict=False)

