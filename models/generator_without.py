import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from math import ceil

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class TreeGCN(nn.Module):
    def __init__(self, depth, features, degrees, support=10, node=1, upsample=False, activation=True):
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth+1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        super(TreeGCN, self).__init__()

        self.W_root = nn.ModuleList([nn.Linear(features[inx], self.out_feature, bias=False) for inx in range(self.depth+1)])

        if self.upsample:
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree*self.in_feature))
        
        self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature*support, bias=False),
                                    nn.Linear(self.in_feature*support, self.out_feature, bias=False))

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()
        self.instance = nn.InstanceNorm1d(features[depth])

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain('relu'))

        stdv = 1. / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        root = 0
        for inx in range(self.depth+1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1,1,repeat_num).view(tree[0].size(0),-1,self.out_feature)

        branch = 0
        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(tree[0].size(0),self.node*self.degree,self.in_feature)
            
            branch = self.W_loop(branch)

            branch = root.repeat(1,1,self.degree).view(tree[0].size(0),-1,self.out_feature) + branch
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch += self.bias.repeat(1,self.node,1)
            branch = self.leaky_relu(branch)
        tree.append(branch)
        return tree


class Generator(nn.Module):
    def __init__(self, features, degrees, support, z_size):
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        
        for inx in range(self.layer_num):
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])
 

    def forward(self, z):
        tree = [z.unsqueeze(1)]
        feat = self.gcn(tree)
        self.pointcloud = feat[-1]

        return self.pointcloud


if __name__ == "__main__":
    generator = Generator([256, 512, 256, 256, 128, 128, 64, 3], [2, 2, 2, 2, 2, 4, 16], 10, 256)
    x = torch.randn(5, 256)
    y = generator(x)
    print(y.size())

