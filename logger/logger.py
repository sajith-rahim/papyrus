import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, config=None, default_level=logging.INFO):
    """
    Setup logging configuration
    """

    if config is None:
        config = 'logger/default.json'

    log_config = Path(config)

    if log_config.is_file():
        config = read_json(log_config)
        # separate streams
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
