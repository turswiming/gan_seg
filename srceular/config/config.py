from omegaconf import OmegaConf

def print_config(config):
    """
    Print the configuration in a readable format.
    """
    print(OmegaConf.to_yaml(config))

def correct_datatype(config):
    if isinstance(config.dataset.KITTISF.fixed_scene_id,int):
        config.dataset.KITTISF.fixed_scene_id = str(config.dataset.KITTISF.fixed_scene_id)
        config.dataset.KITTISF.fixed_scene_id = config.dataset.KITTISF.fixed_scene_id.zfill(6)
    return config



"""
mask2former for point cloud segmentation and scene flow estimation
    config = dict(
        lims=[[-48, 48], [-48, 48], [-3, 1.8]],
        offset=0.5,
        target_scale=1,
        grid_meters=[0.2, 0.2, 0.1],
        scales=[0.5, 1],
        pooling_scale=[0.5, 1, 2, 4, 6, 8, 12],
        sizes=[480, 480, 48],
        n_class=19,
        class_weight=1.0,
        dice_weight=20.0,
        mask_weight=50.0,
        match_class_weight=1.0,
        match_dice_weight=2.0,
        match_mask_weight=5.0,
        num_queries=100,
        dec_layers=6
    )
"""