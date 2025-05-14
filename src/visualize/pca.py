
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def pca(pred_mask, num_components=3):
    #PCA to 3D
    if pred_mask.shape[0]== 3:
        return pred_mask
    if pred_mask.shape[0]< 3:
        # Pad with zeros to make it 3D
        padded_mask = F.pad(pred_mask, (0, 0, 0, 0, 0, 3 - pred_mask.shape[0]))
        return padded_mask
    # Normalize the mask values for better PCA results
    normalized_mask = F.softmax(pred_mask*0.1, dim=1)
    
    # Center the data
    mean = torch.mean(normalized_mask, dim=0, keepdim=True)
    centered_data = normalized_mask - mean
    
    # Compute covariance matrix
    cov = torch.mm(centered_data.t(), centered_data) / (centered_data.size(0) - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    
    # Select top 3 eigenvectors
    top_eigenvectors = eigenvectors[:, :3]
    
    # Project the data onto the top 3 eigenvectors
    color = torch.mm(centered_data, top_eigenvectors)
    
    # Scale to [0, 1] range for visualization
    min_vals = torch.min(color, dim=0, keepdim=True)[0]
    max_vals = torch.max(color, dim=0, keepdim=True)[0]
    color = (color - min_vals) / (max_vals - min_vals + 1e-8)
    return color

