"""
Modifications:

    1. Remove torch.cuda.empty_cache() 
    2. Print time in log
    3. Disable (Comment out) Visdom
    4. Support multiple GPUs
    5. Add learning rate scheduler
    6. Add the ability to load pretrained model
    7. Clip weights of discriminator during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import chain
from torchvision import transforms
from visdom import Visdom
import numpy as np

import time
import datetime
import copy
import yaml
import os, sys
import argparse
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, ".."))

from models.generator import Generator
from models.discriminator import Auxiliary_Discriminator
from models.transfer import Local_Translator_2
from models.encoder import Encoder
from models.gradient_penalty import GradientPenalty_AD
from plyDataloader import plyDataset

from losses.chamfer_loss import ChamferLoss
from losses.earth_mover_distance import EMD

from test import validate

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        print('VisdomLinePlotter')
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y, xlabel='Iterations'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def train(args):
    # -----------------------------------------------File path--------------------------------------------- #
    try:
        os.makedirs(args.weight_path)
    except OSError:
        pass

    try:
        os.makedirs(args.save_img_path)
    except OSError:
        pass



    # ------------------------------------------------Dataset---------------------------------------------- #
    a_train_dataset = plyDataset(root_dir=args.dataset_path, classes=args.class_a_choice, split='train')
    a_train_dataLoader = torch.utils.data.DataLoader(a_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)   
        
    b_train_dataset = plyDataset(root_dir=args.dataset_path, classes=args.class_b_choice, split='train')
    b_train_dataLoader_ = torch.utils.data.DataLoader(b_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)   
    b_train_dataLoader = iter(b_train_dataLoader_)

    a_val_dataset = plyDataset(root_dir=args.val_dataset_path, classes=args.class_a_choice, split="test")
    a_val_dataLoader = torch.utils.data.DataLoader(a_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)   
    b_val_dataset = plyDataset(root_dir=args.val_dataset_path, classes=args.class_b_choice, split="test")
    b_val_dataLoader = torch.utils.data.DataLoader(b_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)   
   
    print("A Training Dataset : {} prepared.".format(len(a_train_dataset)))
    print("A Testing Dataset : {} prepared.".format(len(a_val_dataset)))
    print("B Training Dataset : {} prepared.".format(len(b_train_dataset)))
    print("B Testing Dataset : {} prepared.".format(len(b_val_dataset)))

    # -------------------------------------------------Module---------------------------------------------- #
    generator = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support, z_size=args.z_size).cuda()
    encoder = Encoder(input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, z_size=args.z_size).cuda()
    discriminator = Auxiliary_Discriminator(features=args.D_FEAT).cuda()  
    a2b_transfer = Local_Translator_2().cuda()
    b2a_transfer = Local_Translator_2().cuda()
    GP = GradientPenalty_AD(lambdaGP=args.lambdaGP)
    
    if args.load_pretrained:
        print("Load pretrained weight in {} ...".format(args.pretrained_weight_path))
        checkpoint = torch.load(args.pretrained_weight_path)
        generator.load_state_dict(checkpoint['G_state_dict'])
        encoder.load_state_dict(checkpoint['E_state_dict'])
        discriminator.load_state_dict(checkpoint['D_state_dict'])
        a2b_transfer.load_state_dict(checkpoint['a2b_T_state_dict'])
        b2a_transfer.load_state_dict(checkpoint['b2a_T_state_dict'])

        # Change saved weight and image path
        weight_path_list = args.weight_path.split('/')
        weight_path_list.insert(2, "with_pretrained")
        args.weight_path = os.path.join(*weight_path_list)
        print("New path '{}'to save model weights".format(args.weight_path))
        try:
            os.makedirs(args.weight_path)
        except OSError:
            pass

        save_img_path = args.save_img_path.split('/')
        save_img_path.insert(2, "with_pretrained")
        args.save_img_path = os.path.join(*save_img_path)
        print("New path '{}'to save images".format(args.save_img_path))
        try:
            os.makedirs(args.save_img_path)
        except OSError:
            pass
            
        print("Load pretrained weight: Success")
    optimizerG = optim.Adam(chain(encoder.parameters(), generator.parameters()), lr=args.e_lr, betas=(0, 0.99), weight_decay=args.weight_decay)
    optimizerT = optim.Adam(chain(a2b_transfer.parameters(), b2a_transfer.parameters()), lr=args.t_lr, betas=(0, 0.99), weight_decay=args.weight_decay)
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0, 0.99), weight_decay=args.weight_decay)
    
    if args.is_lr_scheduler_used == True:
        print("Init lr_scheduler")
        batchs_per_epoch=len(a_train_dataLoader)
        print("batchs_per_epoch={}".format(batchs_per_epoch))
        
        lr_scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args.epochs)
        lr_scheduler_T = optim.lr_scheduler.CosineAnnealingLR(optimizerT, T_max=args.epochs)
        lr_scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=args.epochs)

    if args.pretrained_epoch != 0:
        checkpoint_RSCNN = torch.load("./{}/{}.pt".format(args.weight_path, args.pretrained_epoch))
        generator.load_state_dict(checkpoint_RSCNN['G_state_dict'])
        encoder.load_state_dict(checkpoint_RSCNN['E_state_dict'])
        discriminator.load_state_dict(checkpoint_RSCNN['D_state_dict'])
        a2b_transfer.load_state_dict(checkpoint_RSCNN['a2b_T_state_dict'])
        b2a_transfer.load_state_dict(checkpoint_RSCNN['b2a_T_state_dict'])
        optimizerG.load_state_dict(checkpoint_RSCNN['optimizerG'])
        optimizerT.load_state_dict(checkpoint_RSCNN['optimizerT'])
        optimizerD.load_state_dict(checkpoint_RSCNN['optimizerD'])
        lr_scheduler_G.load_state_dict(checkpoint_RSCNN['lr_scheduler_G'])
        lr_scheduler_T.load_state_dict(checkpoint_RSCNN['lr_scheduler_T'])
        lr_scheduler_D.load_state_dict(checkpoint_RSCNN['lr_scheduler_D'])

    chamfer = ChamferLoss().cuda()
    emd = EMD().cuda()
    aux_criterion = nn.CrossEntropyLoss().cuda()
    smoothl1loss = nn.SmoothL1Loss().cuda()

    print("Network prepared.")
    all_a_z = []
    all_b_z = []

    print('Start training...') # @Yung
    start_time = time.time() # @Yung
    # ------------------------------------------------Training--------------------------------------------- #
    for epoch in range(args.pretrained_epoch+1, args.epochs+1):
        for _iter, data in enumerate(a_train_dataLoader, 0):
            # ---------------------- network and data ---------------------- #
            generator.train()
            encoder.train()
            discriminator.train()
            a2b_transfer.train()
            b2a_transfer.train()

            a_points, _ = data
            a_points = Variable(a_points.cuda())
            a_labels = torch.tensor([0] * a_points.size(0)).long().cuda() #[0,...,0]

            try:
                b_points, _ = b_train_dataLoader.next()
            except:
                b_train_dataLoader = iter(b_train_dataLoader_)
                b_points, _ = b_train_dataLoader.next()
            
            b_points = Variable(b_points.cuda())
            b_labels = torch.tensor([1] * b_points.size(0)).long().cuda() #[1,...,1]

            all_labels = torch.cat([a_labels, b_labels], dim=0)
            randn_index = torch.randperm(all_labels.size(0))
            all_labels = all_labels[randn_index]

            # ---------------------- discriminator ---------------------- #
            for _ in range(args.D_iter):
                optimizerD.zero_grad()
                
                with torch.no_grad():
                    all_points = torch.cat([a_points, b_points], dim=0)
                    all_z = encoder(all_points)
                    
                    a_z = all_z[:args.batch_size]
                    b_z = all_z[args.batch_size:]
                    
                    a2b_z = a2b_transfer(a_z)
                    b2a_z = b2a_transfer(b_z)
                    
                    all_fake_z = torch.cat([b2a_z, a2b_z], dim=0)
                    all_fake_z = all_fake_z[randn_index]
                    all_points = all_points[randn_index]
                    fake_points = generator(all_fake_z)

                d_real, real_labels = discriminator(all_points)
                d_fake, fake_labels = discriminator(fake_points)

                d_real = -d_real.mean() + aux_criterion(real_labels, all_labels)
                d_fake = d_fake.mean() + aux_criterion(fake_labels, all_labels)

                gp_loss_fake = GP(discriminator, all_points.data, fake_points.data)

                d_loss = d_real + d_fake + gp_loss_fake
                d_loss.backward()
                optimizerD.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
                    
            # ---------------------- Encoder and generator ---------------------- #
            for _ in range(args.G_iter):
                optimizerG.zero_grad()
                all_points = torch.cat([a_points, b_points], dim=0)
                all_z = encoder(all_points)
                a_z = all_z[:args.batch_size]
                b_z = all_z[args.batch_size:]

                a2b_z = a2b_transfer(a_z)
                b2a_z = b2a_transfer(b_z)
                a2b2a_z = b2a_transfer(a2b_z)
                b2a2b_z = a2b_transfer(b2a_z)

                all_points = all_points[randn_index]
                all_z = all_z[randn_index]
                fake_points = generator(all_z)

                recons_loss = torch.mean(emd(fake_points + 0.5, all_points + 0.5))
            
                all_transfer_z = torch.cat([a2b2a_z, b2a2b_z], dim=0)
                all_transfer_z = all_transfer_z[randn_index]
                transfer_points = generator(all_transfer_z)
                cycle_recons_loss = torch.mean(emd(transfer_points + 0.5, all_points + 0.5))

                g_loss = recons_loss * args.recons_coef + cycle_recons_loss
                g_loss.backward()
                optimizerG.step()

            if len(all_a_z) == len(a_train_dataLoader):
                all_a_z[_iter]=a_z.data
                all_b_z[_iter]=b_z.data
            else:    
                all_a_z.append(a_z.data)
                all_b_z.append(b_z.data)
            # ---------------------- transfer ---------------------- #
            for _ in range(args.T_iter):
                optimizerT.zero_grad()
                a_z = a_z.detach()
                b_z = b_z.detach()

                a2a_z = b2a_transfer(a_z)
                b2b_z = a2b_transfer(b_z)
                a2b_z = a2b_transfer(a_z)
                b2a_z = b2a_transfer(b_z)
                a2b2a_z = b2a_transfer(a2b_z)
                b2a2b_z = a2b_transfer(b2a_z)

                all_transfer_z = torch.cat([b2a_z, a2b_z], dim=0)
                all_transfer_z = all_transfer_z[randn_index]
                
                fake_points = generator(all_transfer_z)
                t_fake, t_fake_labels = discriminator(fake_points)
                t_fake = -t_fake.mean() + aux_criterion(t_fake_labels, all_labels)

                fp_loss = smoothl1loss(a2a_z, a_z) + smoothl1loss(b2b_z, b_z)
                fp_loss += 0.5*(smoothl1loss(a2a_z[:, :128], a_z[:, :128]) + smoothl1loss(b2b_z[:, :128], b_z[:, :128]))
 
                cycle_loss = smoothl1loss(a2b2a_z, a_z) + smoothl1loss(b2b_z, b_z)
                cycle_loss += 0.5*(smoothl1loss(a2b2a_z[:, :128], a_z[:, :128]) + smoothl1loss(b2a2b_z[:, :128], b_z[:, :128]))

                all_a_mean = torch.cat(all_a_z).mean(0)
                all_b_mean = torch.cat(all_b_z).mean(0)

                center_loss = smoothl1loss(a2b_z.mean(0), all_b_mean.detach()) + smoothl1loss(b2a_z.mean(0), all_a_mean.detach())

                a_distance = a_z - all_a_mean
                b_distance = b_z - all_b_mean
                
                a_distance = a_distance.detach()
                b_distance = b_distance.detach()

                
                a2b_distance = a2b_z - all_b_mean
                b2a_distance = b2a_z - all_a_mean


                cp_loss = smoothl1loss(a2b_distance, a_distance) + smoothl1loss(b2a_distance, b_distance)

                t_loss = t_fake + cycle_loss * args.cycle_coef + fp_loss* args.fp_coef + center_loss* args.center_coef + cp_loss * args.cp_coef
                t_loss.backward()
                optimizerT.step()
                

            # --------------------- Visualization -------------------- #
            if _iter % args.log_pre_iter == 0:
                # Time log
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                print("Elapsed time [{}]".format(elapsed),
                      "[Epoch/Iter] ", "{:3}/{:3} | {:3}/{:3}".format(epoch, args.epochs, _iter, len(a_train_dataLoader)),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss.item()), 
                      "[ t_Loss ] ", "{: 7.6f}".format(t_loss.item()),
                      "[ g_Loss ] ", "{: 7.2f}".format(g_loss.item()),
                      "[ recons_Loss ] ", "{: 7.2f}".format(recons_loss.item()), 
                      "[ cycle_recons_loss ] ", "{: 7.2f}".format(cycle_recons_loss.item()), 
                      flush=True)

                # plotter.plot('d loss', 'd real', 'd loss', len(a_train_dataLoader)*epoch+_iter, d_real.item())
                # plotter.plot('d loss', 'd fake', 'd loss', len(a_train_dataLoader)*epoch+_iter, d_fake.item())
                # plotter.plot('d loss', 'discriminator', 'd loss', len(a_train_dataLoader)*epoch+_iter, d_real.item() + d_fake.item())
                # plotter.plot('d loss', 'generator', 'd loss', len(a_train_dataLoader)*epoch+_iter, t_fake.item())
                # plotter.plot('fp_loss', 'train', 'fp_loss', len(a_train_dataLoader)*epoch+_iter, fp_loss.item())
                # plotter.plot('center_loss', 'train', 'center_loss', len(a_train_dataLoader)*epoch+_iter, center_loss.item())
                # plotter.plot('t_loss', 'train', 't_loss', len(a_train_dataLoader)*epoch+_iter, t_loss.item())
                # plotter.plot('cycle_loss', 'train', 'cycle_loss', len(a_train_dataLoader)*epoch+_iter, cycle_loss.item())
                # plotter.plot('cp_loss', 'train', 'cp_loss', len(a_train_dataLoader)*epoch+_iter, cp_loss.item())
                # plotter.plot('g_loss', 'train', 'g_loss', len(a_train_dataLoader)*epoch+_iter, g_loss.item())
                # plotter.plot('cycle_recons_loss', 'train', 'cycle_recons_loss', len(a_train_dataLoader)*epoch+_iter, cycle_recons_loss.item())
                # plotter.plot('recons_loss', 'a+b', 'recons_loss', len(a_train_dataLoader)*epoch+_iter, recons_loss.item())
                
        if args.is_lr_scheduler_used == True:
            lr_scheduler_G.step()
            lr_scheduler_T.step()
            lr_scheduler_D.step()
            # plotter.plot('lr_scheduler', 'G', 'lr_scheduler', epoch, optimizerG.param_groups[0]["lr"], xlabel='Epochs')
            # plotter.plot('lr_scheduler', 'T', 'lr_scheduler', epoch, optimizerT.param_groups[0]["lr"], xlabel='Epochs')
            # plotter.plot('lr_scheduler', 'D', 'lr_scheduler', epoch, optimizerD.param_groups[0]["lr"], xlabel='Epochs')

        if epoch % args.save_pre_epoch == 0:

            torch.save({
                        'E_state_dict': encoder.state_dict(),
                        'D_state_dict': discriminator.state_dict(),
                        'G_state_dict': generator.state_dict(),
                        'a2b_T_state_dict': a2b_transfer.state_dict(),
                        'b2a_T_state_dict': b2a_transfer.state_dict(),
                        'optimizerG': optimizerG.state_dict(),
                        'optimizerT': optimizerT.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        'lr_scheduler_G': lr_scheduler_G.state_dict(),
                        'lr_scheduler_T': lr_scheduler_T.state_dict(),
                        'lr_scheduler_D': lr_scheduler_D.state_dict()
                }, '{}/{}.pt'.format(args.weight_path, epoch))
            
            validate(a_val_dataLoader, b_val_dataLoader, encoder, a2b_transfer, b2a_transfer, generator, args, epoch)
            print('Checkpoint is saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config", default="configs/config_chair_table_clip_Wcenter5_Wcp5_Wcycle10_lrScheduler_weightDecay_lrTe4.yaml", type=str
    )
    parser.add_argument(
    "--gpu", default=0, type=int  
    )
    parser.add_argument(
    "--load_pretrained", default=0, type=int
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICE']= str(args.gpu)
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config["common"].items():
        setattr(args, k, v)
    for k, v in config["train"].items():
        setattr(args, k, v)
    print(args)
    # Plots
    # global plotter
    # plotter = VisdomLinePlotter()
    train(args)