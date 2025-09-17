"""
Visualization utility functions for instance segmentation and point cloud visualization.

This module provides utility functions for:
- Creating colormaps for instance visualization
- Converting instance masks to colors
- Remapping instance labels for visualization
"""

import torch
import torch.nn.functional as F
import numpy as np

def remap_instance_labels(labels):
    """
    Remap arbitrary integer instance labels to consecutive numbers starting from 0.
    
    This function takes instance labels that may have gaps or arbitrary values and
    remaps them to a consecutive sequence starting from 0. For example:
    Input labels [0,1,8,1] would be remapped to [0,1,2,1]
    
    Args:
        labels (torch.Tensor): Input tensor containing instance labels with arbitrary values
    
    Returns:
        torch.Tensor: New tensor with remapped consecutive instance labels
    """
    unique_labels = torch.unique(labels)
    mapping = {label.item(): idx for idx, label in enumerate(sorted(unique_labels))}
    # print(f"remap {mapping}")
    # Create new label tensor
    remapped = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        remapped[labels == old_label] = new_label
        
    return remapped

def create_label_colormap():
    """
    Create a predefined colormap for instance segmentation visualization.
    
    This function creates a colormap similar to the one used in CITYSCAPES segmentation
    benchmark, providing distinct colors for up to 70 different instances. The colors
    are chosen to be visually distinguishable from each other.
    
    Returns:
        torch.Tensor: RGB colormap tensor with shape [256, 3] containing integer RGB values
                     in range [0, 255]
    """
    colormap = np.zeros((256, 3), dtype=np.int64)
    colormap[0] = [100, 134, 102]
    colormap[1] = [166, 206, 227]
    colormap[2] = [31, 120, 180]
    colormap[3] = [178, 223, 138]
    colormap[4] = [51, 160, 44]
    colormap[5] = [251, 154, 153]
    colormap[6] = [227, 26, 28]
    colormap[7] = [253, 191, 111]
    colormap[8] = [255, 127, 0]
    colormap[9] = [202, 178, 214]
    colormap[10] = [106, 61, 154]
    colormap[11] = [255, 255, 153]
    colormap[12] = [177, 89, 40]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    colormap[19] = [100, 134, 102]
    colormap[20] = [166, 206, 227]
    colormap[21] = [31, 120, 180]
    colormap[22] = [178, 223, 138]
    colormap[23] = [51, 160, 44]
    colormap[24] = [251, 154, 153]
    colormap[25] = [227, 26, 28]
    colormap[26] = [253, 191, 111]
    colormap[27] = [255, 127, 0]
    colormap[28] = [202, 178, 214]
    colormap[29] = [106, 61, 154]
    colormap[30] = [255, 255, 153]
    colormap[31] = [177, 89, 40]
    colormap[32] = [0, 0, 142]
    colormap[33] = [0, 0, 70]
    colormap[34] = [0, 60, 100]
    colormap[35] = [0, 80, 100]
    colormap[36] = [0, 0, 230]
    colormap[37] = [119, 11, 32]
    colormap[38] = [100, 134, 102]
    colormap[39] = [166, 206, 227]
    colormap[40] = [31, 120, 180]
    colormap[41] = [178, 223, 138]
    colormap[42] = [51, 160, 44]
    colormap[43] = [251, 154, 153]
    colormap[44] = [227, 26, 28]
    colormap[45] = [253, 191, 111]
    colormap[46] = [255, 127, 0]
    colormap[47] = [202, 178, 214]
    colormap[48] = [106, 61, 154]
    colormap[49] = [255, 255, 153]
    colormap[50] = [177, 89, 40]
    colormap[51] = [0, 0, 142]
    colormap[52] = [0, 0, 70]
    colormap[53] = [0, 60, 100]
    colormap[54] = [0, 80, 100]
    colormap[55] = [0, 0, 230]
    colormap[56] = [119, 11, 32]
    colormap[57] = [100, 134, 102]
    colormap[58] = [166, 206, 227]
    colormap[59] = [31, 120, 180]
    colormap[60] = [178, 223, 138]
    colormap[61] = [51, 160, 44]
    colormap[62] = [251, 154, 153]
    colormap[63] = [227, 26, 28]
    colormap[64] = [253, 191, 111]
    colormap[65] = [255, 127, 0]
    colormap[66] = [202, 178, 214]
    colormap[67] = [106, 61, 154]
    colormap[68] = [255, 255, 153]
    colormap[69] = [177, 89, 40]
    colormap[70] = [0, 0, 142]

    return torch.from_numpy(colormap).long()

def color_mask(mask):
    """
    Convert instance masks to RGB colors for visualization using a predefined colormap.
    
    This function maps each instance in the mask to a unique RGB color from a predefined
    CITYSCAPES-style colormap. The colors are normalized to [0,1] range for visualization.
    
    Args:
        mask (torch.Tensor): Instance mask tensor with shape [K, N], where K is number
                            of instances and N is number of points
        
    Returns:
        torch.Tensor: RGB color values for each point with shape [N, 3], normalized to [0,1]
    """
    color_label = create_label_colormap()
    #get all different values in mask
    mask_argmax = torch.argmax(mask, dim=0)
    unique_values = torch.unique(mask_argmax)
    color_result = torch.zeros((mask_argmax.shape[0], 3), dtype=torch.float32)
    for i in range(len(unique_values)):
        # Convert color_label to float before assignment
        color_result[mask_argmax == unique_values[i]] = color_label[unique_values[i]].to(torch.float32)
    color_result = color_result / 255.0
    return color_result 