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