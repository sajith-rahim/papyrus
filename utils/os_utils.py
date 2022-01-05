import os
from collections import OrderedDict
from os import path as osp
import shutil
from pathlib import Path
import json


def remove_if_path_exists(path):
    """ removes directory given in the path if it exists """
    if osp.exists(path):
        shutil.rmtree(path)


def ensure_dir(directory):
    """ creates directory if it doesn't exist """
    directory = Path(directory)
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=False)
        return False
    return True


def remove_if_exists(directory):
    """ removes directory if it exists """
    directory = Path(directory)
    if osp.exists(directory):
        shutil.rmtree(directory)


def read_json(file):
    """ read json """
    file = Path(file)
    with file.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, file):
    """ write json """
    file = Path(file)
    with file.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


# os-env utils

def get_env(env_name: str, default= None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.
    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file= None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    raise NotImplementedError
    #dotenv.load_dotenv(dotenv_path=env_file, override=True)

