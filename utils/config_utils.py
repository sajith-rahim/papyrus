from omegaconf import OmegaConf

def merge_config(config, cli_conf_update):
    r"""Merge configs.
    Update predefined conf values with kwargs
    Arguments:
        config
            predefined conf
        cli_conf_update
             kwargs
    """
    conf = OmegaConf.merge(config, cli_conf_update)
    return conf