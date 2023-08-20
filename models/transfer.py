import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import pointnet2_utils


def normalize_batch(points):
    bb_max = points.max(1)[0]
    bb_min = points.min(1)[0]
    length = (bb_max - bb_min)
    mean = (bb_max + bb_min) / 2.0
    points = (points - mean.unsqueeze(1)) /length.unsqueeze(1)
    return points, mean, length


class Translator(nn.Module):
    def __init__(self, input_size=256, output_size=256):
        super(Translator, self).__init__()
        self.translator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, z):
        z = self.translator(z)
        return z


class Auxiliary_Discriminator(nn.Module):
    def __init__(self):
        super(Auxiliary_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc_dis = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.fc_aux = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, z):
        out = self.discriminator(z)
        realfake = self.fc_dis(out)
        labels = self.fc_aux(out)
        return realfake, labels


class Local_Translator_1(nn.Module):
    def __init__(self):
        super(Local_Translator_1, self).__init__()
        self.translator_modules = nn.ModuleList()
        for _ in range(4):
            self.translator_modules.append(Translator(input_size=64, output_size=64))
       
        

    def forward(self, z):
        output = []
        for i, module in enumerate(self.translator_modules):
            output.append(module(z[:, i*64:(i+1)*64]))
        output = torch.cat(output, dim=1)
        return output


class Local_Translator_2(nn.Module):
    def __init__(self):
        super(Local_Translator_2, self).__init__()
        self.translator_modules = nn.ModuleList()
        for i in range(4):
            self.translator_modules.append(Translator(input_size=(i+1)*64, output_size=64))
       
    def forward(self, z):
        output = []
        for i, module in enumerate(self.translator_modules):
            output.append(module(z[:, :(i+1)*64]))
        output = torch.cat(output, dim=1)
        return output

if __name__ == "__main__":
    x = torch.randn(5, 256)
    transfer = Translator()
    num = sum(p.numel() for p in transfer.parameters())
    print(num*2)

    y = transfer(x)
    print(y.size())
