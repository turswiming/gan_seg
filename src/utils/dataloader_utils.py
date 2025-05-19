import torch
from dataset.av2_dataset import AV2Dataset
from dataset.per_scene_dataset import PerSceneDataset

def infinite_dataloader(dataloader):
    """
    Creates an infinite iterator that cycles through the dataset indefinitely.
    
    This function wraps a PyTorch DataLoader to create an infinite stream of batches.
    When the end of the dataset is reached, it automatically restarts from the beginning.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The original PyTorch DataLoader to wrap
        
    Yields:
        dict: A batch of data containing:
            - point_cloud_first (torch.Tensor): First point cloud [B, N, 3]
            - point_cloud_second (torch.Tensor): Second point cloud [B, N, 3]
            - flow (torch.Tensor): Ground truth flow vectors [B, N, 3]
    """
    while True:
        for batch in dataloader:
            yield batch

def create_dataloaders(config):
    """
    Creates and initializes data loaders based on the provided configuration.
    
    This function handles the creation of both the standard DataLoader and an infinite
    version. It also determines the batch size and number of points per sample from
    the first batch.
    
    Args:
        config: Configuration object containing:
            - dataset.name (str): Name of the dataset ("AV2" or "MOVI_F")
            - dataloader.batchsize (int): Number of samples per batch
            
    Returns:
        tuple: A tuple containing:
            - dataloader (torch.utils.data.DataLoader): Standard PyTorch DataLoader
            - infinite_loader (generator): Infinite version of the DataLoader
            - batch_size (int): Actual batch size from the first batch
            - N (int): Number of points per sample
            
    Raises:
        ValueError: If the specified dataset name is not supported
    """
    # Create dataset based on config
    if config.dataset.name == "AV2":
        dataset = AV2Dataset()
    elif config.dataset.name == "MOVI_F":
        dataset = PerSceneDataset()
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported")
    
    # Create dataloader with batch dimension handling
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=True,
        collate_fn=lambda batch: {
            "point_cloud_first": torch.stack([item["point_cloud_first"] for item in batch]),
            "point_cloud_second": torch.stack([item["point_cloud_second"] for item in batch]),
            "flow": torch.stack([item["flow"] for item in batch])
        }
    )
    
    # Create infinite dataloader
    infinite_loader = infinite_dataloader(dataloader)
    
    # Get sample to determine dimensions
    sample = next(infinite_dataloader(dataloader))
    batch_size, N, _ = sample["point_cloud_first"].shape
    
    return dataloader, infinite_loader, batch_size, N 