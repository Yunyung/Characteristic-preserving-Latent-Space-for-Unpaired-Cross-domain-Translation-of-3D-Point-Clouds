import os
import numpy as np
import torch

import pandas as pd
from torch.utils.data import Dataset
from pyntcloud import PyntCloud

from utils.plot_utils import plot_3d_point_cloud
import matplotlib.pyplot as plt

class plyDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/shapenet', classes="chair",
                 transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.classes = classes
        self.pc_names = sorted(os.listdir(os.path.join(root_dir, f"{classes}_{split}")))


    def __len__(self):
        return len(self.pc_names)

    def __getitem__(self, idx):

        pc_filename = self.pc_names[idx]
        pc_filepath = os.path.join(self.root_dir, f"{self.classes}_{self.split}", pc_filename)
        pc = PyntCloud.from_file(pc_filepath)
        pc = np.array(pc.points)
        if self.transform:
            pc = self.transform(pc)

        return pc[:, :3], pc_filename


"""
@yung
"""
class plyDataset_flexible(Dataset):
    def __init__(self, root_dir='/home/datasets/shapenet', classes="chair",
                 transform=None, split=None, fiter_out_dir=True):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.classes = classes
        if split is None:
            self.pc_names = []

            if fiter_out_dir == True:
                for pc_names in os.listdir(root_dir):
                    path = os.path.join(root_dir, pc_names)
                    if os.path.isdir(path):
                        print(f'Filter out direcotry: "{pc_names}"')
                        continue
                    else:
                        self.pc_names.append(pc_names)
                self.pc_names = sorted(self.pc_names)
            else:
                self.pc_names = sorted(os.listdir(root_dir))
            
        else:
            self.pc_names = sorted(os.listdir(os.path.join(root_dir, f"{classes}_{split}")))

        
    def __len__(self):
        return len(self.pc_names)

    def __getitem__(self, idx):

        pc_filename = self.pc_names[idx]

        if self.split is None:
            pc_filepath = os.path.join(self.root_dir, pc_filename)
        else:
            pc_filepath = os.path.join(self.root_dir, f"{self.classes}_{self.split}", pc_filename)
            
        pc = PyntCloud.from_file(pc_filepath)
        pc = np.array(pc.points)
        if self.transform:
            pc = self.transform(pc)

        return pc[:, :3], pc_filename
