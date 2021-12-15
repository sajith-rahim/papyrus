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
