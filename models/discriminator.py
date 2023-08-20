import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from math import ceil

class Auxiliary_Discriminator(nn.Module):
    def __init__(self, features, num_classes=2):
        self.layer_num = len(features)-1
        super(Auxiliary_Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
                                         nn.Linear(features[-1], features[-2]),
                                         nn.Linear(features[-2], features[-2])
                                        )
        self.fc_dis = nn.Sequential(
            nn.Linear(features[-2], features[-2]),
            nn.Linear(features[-2], 1),

        )
        self.fc_aux = nn.Sequential(
            nn.Linear(features[-2], features[-2]),
            nn.Linear(features[-2], num_classes),
        )

    def forward(self, f):
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out) # (B, 1)
        realfake = self.fc_dis(out)
        classes = self.fc_aux(out)
        return realfake, classes


if __name__ == "__main__":
    discriminator = Auxiliary_Discriminator([3,  64,  128, 256, 512]).cuda()


    num = sum(p.numel() for p in discriminator.parameters())
    print(num)


    # z = torch.randn(5, 256).cuda()
    # s=[torch.randn(5, 64).cuda(), torch.randn(5, 64).cuda(), torch.randn(5, 64).cuda(), torch.randn(5, 64).cuda()]
    # pred = generator(z, s)
    # print(pred.size())

    x = torch.randn(5, 2048, 3).cuda()
    _, pred_label = discriminator(x)
    print(pred_label)

