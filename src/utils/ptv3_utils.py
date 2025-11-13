"""
Utilities for loading PointTransformerV3 pretrained models.
"""

import os
import torch
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib


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
        checkpoint = torch.load(pretrained_path, map_location='cpu')
    elif pretrained_name:
        if pretrained_name.lower() == 'sonata':
            checkpoint = download_sonata_weights()
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
        if missing_keys:
            print(f"Missing backbone keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected backbone keys: {len(unexpected_keys)}")
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
            print(f"Warning: Could not load weights directly: {e}")
            print("Model will use random initialization")
    
    return model


def download_sonata_weights(cache_dir=None):
    """
    Download Sonata self-supervised pretrained weights.
    Sonata is available at: https://github.com/facebookresearch/sonata
    
    Args:
        cache_dir: Directory to cache downloaded weights
    
    Returns:
        checkpoint: Loaded checkpoint dict
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/ptv3_weights")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sonata repository URLs (check official repo for latest links)
    sonata_urls = [
        "https://github.com/facebookresearch/sonata/releases/download/v1.0/ptv3_sonata.pth",
        "https://dl.fbaipublicfiles.com/sonata/ptv3_sonata.pth",
    ]
    
    checkpoint_path = os.path.join(cache_dir, "sonata_ptv3.pth")
    
    if not os.path.exists(checkpoint_path):
        print("Downloading Sonata self-supervised pretrained model...")
        print("Note: If download fails, please download manually from:")
        print("  https://github.com/facebookresearch/sonata")
        
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
                "Please download manually from https://github.com/facebookresearch/sonata "
                "and place the checkpoint file at the expected location."
            )
    
    print(f"Loading Sonata checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


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
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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

