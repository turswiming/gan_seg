import torch
from dataset.av2_dataset import AV2PerSceneDataset,AV2SequenceDataset
from dataset.movi_per_scene_dataset import MOVIPerSceneDataset
from dataset.kitti_dataset import KITTIPerSceneDataset
def infinite_dataloader(dataloader):
    """
    Create an infinite iterator that cycles through a dataloader indefinitely.
    
    This function is useful for training scenarios where you want to continue
    training beyond one epoch without manual epoch handling. It will continuously
    cycle through the dataset, allowing for unlimited iterations.
    
    Args:
        dataloader (DataLoader): The source PyTorch DataLoader to create an infinite
                               iterator from
    
    Returns:
        iterator: An infinite iterator that yields batches from the dataloader
                 indefinitely by cycling through the dataset
    
    Example:
        >>> train_loader = DataLoader(dataset, batch_size=32)
        >>> infinite_train_loader = infinite_dataloader(train_loader)
        >>> for batch in infinite_train_loader:  # This loop will never end
        >>>     # Process batch
        >>>     pass
    """
    while True:
        for batch in dataloader:
            yield batch

def create_dataloaders(config):
    """
    Create training and infinite dataloaders based on configuration.
    
    Args:
        config (OmegaConf): Configuration object containing dataset and dataloader parameters
                           including batch_size, num_workers, and dataset-specific settings
    
    Returns:
        tuple: A tuple containing:
            - dataloader (DataLoader): The main PyTorch dataloader
            - infinite_loader (iterator): An infinite iterator cycling through the dataset
            - batch_size (int): The batch size used for the dataloaders
            - N (int): Total number of samples in the dataset
            
    Note:
        The infinite loader is particularly useful for training scenarios where
        you want to continue training beyond one epoch without manual epoch handling.
    """
    # Create dataset based on config
    if config.dataset.name == "AV2":
        dataset = AV2PerSceneDataset()
    elif config.dataset.name == "AV2Sequence":
        dataset = AV2SequenceDataset(max_k=3)
    elif config.dataset.name == "MOVI_F":
        dataset = MOVIPerSceneDataset()
    elif config.dataset.name == "KITTISF":
        if config.dataset.KITTISF.fixed_scene_id is not None:
            dataset = KITTIPerSceneDataset(
                data_root=config.dataset.KITTISF.data_root, 
                downsampled=config.dataset.KITTISF.downsampled, 
                fixed_scene_id=config.dataset.KITTISF.fixed_scene_id
                )
        else:
            dataset = KITTIPerSceneDataset(
                data_root=config.dataset.KITTISF.data_root, 
                downsampled=config.dataset.KITTISF.downsampled
                )
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported")
    if config.dataset.val_name == "AV2Sequence":
        val_dataset = AV2SequenceDataset(max_k=3)
    elif config.dataset.val_name == "AV2Sequence_val":
        val_dataset = AV2SequenceDataset(max_k=1)
    elif config.dataset.val_name == "KITTISF":
        val_dataset = KITTIPerSceneDataset(data_root=config.dataset.KITTISF.data_root, downsampled=config.dataset.KITTISF.downsampled, fixed_scene_id=config.dataset.KITTISF.fixed_scene_id)
    else:
        raise ValueError(f"Dataset {config.dataset.val_name} not supported")
    # Create dataloader with batch dimension handling
    collate_fn_lambda = lambda batch: {
            "point_cloud_first": [item["point_cloud_first"] for item in batch if "point_cloud_first" in item],
            "point_cloud_second": [item["point_cloud_second"] for item in batch if "point_cloud_second" in item],
            "flow": [item["flow"] for item in batch if "flow" in item],
            "dynamic_instance_mask": [item["dynamic_instance_mask"] for item in batch if "dynamic_instance_mask" in item],
            "background_static_mask": [item["background_static_mask"] for item in batch if "background_static_mask" in item],
            "foreground_static_mask": [item["foreground_static_mask"] for item in batch if "foreground_static_mask" in item],
            "foreground_dynamic_mask": [item["foreground_dynamic_mask"] for item in batch if "foreground_dynamic_mask" in item],
            "idx": [item["idx"] for item in batch if "idx" in item],
            "idx2": [item["idx2"] for item in batch if "idx2" in item],
            "total_frames": [item["total_frames"] for item in batch if "total_frames" in item],
            "self": [item["self"] for item in batch if "self" in item],
            "k": [item["k"] for item in batch if "k" in item],
            "ego_motion": [item["ego_motion"] for item in batch if "ego_motion" in item],
        }
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn_lambda,
        pin_memory=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn_lambda,
        pin_memory=False
    )
    # Create infinite dataloader
    infinite_loader = infinite_dataloader(dataloader)
    
    # Get sample to determine dimensions
    sample = next(infinite_dataloader(dataloader))
    batch_size = len(sample["point_cloud_first"])
    N = sample["point_cloud_first"][0].shape[0]
    
    return dataloader, infinite_loader, val_dataloader, batch_size, N 