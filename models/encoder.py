import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils")) 

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from point2_modules import RSCNN_module
import pytorch_utils as pt_utils

class Encoder_with_convs_and_symmetry(nn.Module):
    def __init__(self, input_size):
        super(Encoder_with_convs_and_symmetry, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_size, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
        )

    def forward(self, feature):
        feature = self.layer(feature)
        feature = feature.max(2)[0]
        return feature



class Encoder(nn.Module):
    """
        
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=0, relation_prior=1, use_xyz=True, z_size=256):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.conv_layer = nn.ModuleList()
        
        self.SA_modules.append(
            RSCNN_module(
                npoint=512,
                radii=[0.1],
                nsamples=[64],
                mlps=[[input_channels, 128]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            RSCNN_module(
                npoint=256,
                radii=[0.2],
                nsamples=[64],
                mlps=[[128, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            RSCNN_module(
                npoint=128,
                radii=[0.3],
                nsamples=[64],
                mlps=[[256, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        
        self.SA_modules.append(
            RSCNN_module(
                npoint=64,
                radii=[0.4],
                nsamples=[64],
                mlps=[[256, z_size]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.conv_layer.append(
            Encoder_with_convs_and_symmetry(128)
        )
        self.conv_layer.append(
            Encoder_with_convs_and_symmetry(256)
        )
        self.conv_layer.append(
            Encoder_with_convs_and_symmetry(256)
        )
        self.conv_layer.append(
            Encoder_with_convs_and_symmetry(z_size)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        z = []
        for module_i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
            z.append(self.conv_layer[module_i](features))
        z = torch.cat(z, dim=1)
        return z


    def local_feat(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        all_xyz = []
        all_features = []
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
            all_xyz.append(xyz)
            all_features.append(features.max(2)[0])
            # all_features.append(features)
        return all_xyz, all_features


if __name__ == "__main__":
    rscnn = Encoder().cuda()
    num = sum(p.numel() for p in rscnn.parameters())
    print(num)

    x = torch.randn((5, 2048, 3)).cuda()
    # z = rscnn(x)
    # print(z.size())

    local_xyz, local_feats = rscnn.local_feat(x)
    print(len(local_feats))
    for local_feat in local_feats:
        print(local_feat.size())
    
    # for xyz in local_xyz:
    #     print(xyz.size())
    # content_points = torch.randn((1, 2048, 3)).cuda()
    # style_points = torch.randn((1, 2048, 3)).cuda()
    # z = rscnn.wct(content_points, style_points)
    # print(z.size())
