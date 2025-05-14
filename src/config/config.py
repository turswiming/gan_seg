from omegaconf import OmegaConf

def print_config(config):
    """
    Print the configuration in a readable format.
    """
    print(OmegaConf.to_yaml(config))