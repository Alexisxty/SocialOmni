import argparse
import os
import os.path as osp
from mmengine.config import Config


def parse_config(path=None):
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str, default="config/bizgen_base.py")
        args = parser.parse_args()
        path = args.config_dir
    config = Config.fromfile(path)
    
    config.config_dir = path
        
    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        config.local_rank = -1

    return config

