from dataclasses import dataclass

from config import Paths,Files,Params


@dataclass
class MNISTConfig:
    paths: Paths
    files: Files
    params: Params
