import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import argparse
import PIL
import yaml
from visdom import Visdom
from sklearn import manifold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, ".."))
from utils.plot_utils import plot_3d_point_cloud
from models.generator import Generator
from models.generator_without import Generator as Generator_without
from models.discriminator import Auxiliary_Discriminator
from models.encoder import Encoder
from models.encoder_without import Encoder as Encoder_without
from models.transfer import Translator, Local_Translator_2

from plyDataloader import plyDataset
sys.path.append(os.path.join(BASE_DIR, "../../losses"))
from chamfer_loss import ChamferLoss
from earth_mover_distance import EMD

def undo_normalize(points, mean, length):
    points = points * length.view(length.size(0), 1, 1)
    points = points + mean.unsqueeze(1)

    return points


def normalize_batch(points):
    bb_max = points.max(1)[0]
    bb_min = points.min(1)[0]
    length = (bb_max - bb_min).max(1)[0]
    mean = (bb_max + bb_min) / 2.0
    points = (points - mean.unsqueeze(1)) /length.view(length.size(0), 1, 1)
    return points, mean, length


def scale_aug(points):
    bb_max = points.max(1)[0]
    bb_min = points.min(1)[0]

    mean = (bb_max + bb_min) / 2.0
    scale = torch.FloatTensor(np.random.uniform(0.7, 1, points.size(0)* 3)).cuda()
    scale = scale.view(points.size(0), 3)
    points = (points - mean.unsqueeze(1))
    points = points * scale.unsqueeze(1)
    bb_max = points.max(1)[0]
    bb_min = points.min(1)[0]

    length = (bb_max - bb_min).max(1)[0]
    points /= length.view(length.size(0), 1, 1)

    return points

def test(args):
    # -----------------------------------------------File path--------------------------------------------- #

    try:
        os.makedirs(args.save_img_path)
    except OSError:
        pass

    try:
        os.makedirs(args.save_a_img_path)
    except OSError:
        pass

    try:
        os.makedirs(args.save_b_img_path)
    except OSError:
        pass

    try:
        os.makedirs(args.plot_npy_path)
    except OSError:
        pass

    try:
        os.makedirs(args.plot_xml_path)
    except OSError:
        pass
    
    try:
        os.makedirs(args.plot_jpg_path)
    except OSError:
        pass


    # ----------------------------------------------------------------------------------------------------- #

    # ------------------------------------------------Dataset---------------------------------------------- #
    a_test_dataset = plyDataset(root_dir=args.dataset_path, classes=args.class_a_choice, split=args.split)
    a_test_dataLoader = torch.utils.data.DataLoader(a_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=args.drop_last)   

    b_test_dataset = plyDataset(root_dir=args.dataset_path, classes=args.class_b_choice, split=args.split)
    b_test_dataLoader = torch.utils.data.DataLoader(b_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=args.drop_last)   
  

    print("Chair Testing Dataset : {} prepared.".format(len(a_test_dataset)))
    print("Table Testing Dataset : {} prepared.".format(len(b_test_dataset)))
    # ----------------------------------------------------------------------------------------------------- #

    # -------------------------------------------------Module---------------------------------------------- #
    discriminator = Auxiliary_Discriminator(features=args.D_FEAT).cuda() 
    
    if args.G == "with":
        generator = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support, z_size=args.z_size).cuda()
    elif args.G == "without":
        generator = Generator_without(features=args.G_FEAT, degrees=args.DEGREE, support=args.support, z_size=args.z_size).cuda()
    
    if args.E == "with":
        encoder = Encoder(input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, z_size=args.z_size).cuda()
    elif args.E == "without":
        encoder = Encoder_without(input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, z_size=args.z_size).cuda()

    if args.T == "single":
        a2b_transfer = Translator().cuda() 
        b2a_transfer = Translator().cuda() 
    elif args.T == "multi":
        a2b_transfer = Local_Translator_2().cuda() 
        b2a_transfer = Local_Translator_2().cuda() 

    
    checkpoint_RSCNN = torch.load(f"./{args.weight_path}/{args.autoencoder_epoch}.pt")
    encoder.load_state_dict(checkpoint_RSCNN['E_state_dict'])
    generator.load_state_dict(checkpoint_RSCNN['G_state_dict'])
    # discriminator.load_state_dict(checkpoint_RSCNN['D_state_dict'])
    a2b_transfer.load_state_dict(checkpoint_RSCNN['a2b_T_state_dict'])
    b2a_transfer.load_state_dict(checkpoint_RSCNN['b2a_T_state_dict'])

    print("Network prepared.")
    # ---------validate------------------------ #
    # validate(a_test_dataLoader, encoder, generator, args, 0)
    
    # ---------CD EMD------------------------ #
    # calculate_chamfer_emd(a_test_dataLoader, b_test_dataLoader, encoder, generator, args)
    
    # ---------save images------------------------ #
    # save_all_images(a_test_dataLoader, b_test_dataLoader, encoder, generator, a2b_transfer, b2a_transfer, args)
    # save_shape_style_mixing_images(a_test_dataLoader, encoder, generator, args.save_a_img_path, args)
    # save_shape_style_mixing_images(b_test_dataLoader, encoder, generator, args.save_b_img_path, args)
    # save_MVS_images(a_test_dataLoader, b_test_dataLoader, encoder, generator, args)
    # save_MVS_images(b_test_dataLoader, a_test_dataLoader, encoder, generator, args)
    
    # ---------tsne------------------------ #
    # plot_tsne(a_test_dataLoader, b_test_dataLoader, encoder, a2b_transfer, b2a_transfer, generator, args)
    
    # ---------save npy file------------------------ #
    # save_npy_index(a_test_dataset, encoder, a2b_transfer, generator, args)
    # save_npy_index(b_test_dataset, encoder, b2a_transfer, generator, args)
    # save_shape_style_mixing_npy(a_test_dataset, encoder, generator, args)
    # save_shape_style_mixing_npy(b_test_dataset, encoder, generator, args)
    # save_MVS_npy(a_test_dataLoader, b_test_dataLoader, encoder, generator, args)
    # save_MVS_npy(b_test_dataLoader, a_test_dataLoader, encoder, generator, args)



def calculate_chamfer_emd(a_test_dataLoader, b_test_dataloader_, encoder, generator, args):
    
    chamfer = ChamferLoss().cuda()
    emd = EMD().cuda()

    b_test_dataloader = iter(b_test_dataloader_)
    
    chamfer_loss = 0.0
    emd_loss = 0.0

    for _iter, data in enumerate(a_test_dataLoader, 0):
        encoder.eval()
        generator.eval()
        
        a_points, _ = data
        a_points = a_points.cuda().float()
        b_points, _ = b_test_dataloader.next()
        b_points = b_points.cuda().float()
        
        with torch.no_grad():
            a_z = encoder(a_points)
            b_z = encoder(b_points)
            fake_b_points = generator(b_z)
            fake_a_points = generator(a_z)

        chamfer_loss += chamfer(a_points, fake_a_points)
        chamfer_loss += chamfer(b_points, fake_b_points)
        emd_loss += torch.mean(emd(a_points, fake_a_points))/args.point_num
        emd_loss += torch.mean(emd(b_points, fake_b_points))/args.point_num

    print("CD")
    print(chamfer_loss/(2*(_iter+1)*args.batch_size))
    print("EMD")
    print(emd_loss/(2*(_iter+1)))

    print("CD (scale of paper)")
    print(chamfer_loss/(2*(_iter+1)*args.batch_size)/args.point_num * 10000)
    print("EMD (scale of paper)")
    print(emd_loss/(2*(_iter+1)) * 100)


def save_npy_index(a_test_dataset, encoder, transfer, generator, args):
    a_points = []
    a_names = []
    for ps in args.points_index:
        a_point, a_name = a_test_dataset[ps]
        a_point = torch.tensor(a_point).cuda()
        a_points.append(a_point.unsqueeze(0))
        a_names.append(a_name)

    a_points = torch.cat(a_points, dim=0)
    encoder.eval()
    transfer.eval()
    generator.eval()
    with torch.no_grad():
        a_z = encoder(a_points)
        a_recons_points = generator(a_z)
        a2b_z = transfer(a_z)
        b_fake_points = generator(a2b_z)
        
    a_points = a_points.detach().cpu().numpy()
    a_recons_points = a_recons_points.detach().cpu().numpy()
    b_fake_points = b_fake_points.detach().cpu().numpy()
    for k in range(a_points.shape[0]):
        all_points = np.concatenate((a_points[k], a_recons_points[k], b_fake_points[k]), axis=0)
        np.save(f"{args.plot_npy_path}/{args.points_index[k]}_{a_names[k].split('.')[0]}.npy", all_points)


def validate(a_test_dataloader, b_test_dataloader_, encoder, a2b_transfer, b2a_transfer, generator, args, epoch):
    b_test_dataloader = iter(b_test_dataloader_)
    for _iter, data in enumerate(a_test_dataloader, 0):
        encoder.eval()
        generator.eval()
        a2b_transfer.eval()
        b2a_transfer.eval()
        
        a_points, _ = data
        a_points = a_points.cuda()
        
        b_points, _ = b_test_dataloader.next()
        b_points = b_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            a_z = encoder(a_points)
            b_z = encoder(b_points)
            b2a_z = b2a_transfer(b_z)
            a2b_z = a2b_transfer(a_z)
            b_fake_points = generator(a2b_z)
            a_fake_points = generator(b2a_z)
            a_recons_points = generator(a_z)
            b_recons_points = generator(b_z)

       
        a_points = a_points.detach().cpu().numpy()
        b_points = b_points.detach().cpu().numpy()
        a_fake_points = a_fake_points.detach().cpu().numpy()
        b_fake_points = b_fake_points.detach().cpu().numpy()
        a_recons_points = a_recons_points.detach().cpu().numpy()
        b_recons_points = b_recons_points.detach().cpu().numpy()

        
        fig = plt.figure(figsize=(30, 40))
        for k in range(8):
            _ = plot_3d_point_cloud(
                a_points[k][:, 0],
                a_points[k][:, 1],
                a_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_a_choice}_real",
                axis=fig.add_subplot(10, 6, k*6+1, projection='3d')
            )
            _ = plot_3d_point_cloud(
                a_recons_points[k][:, 0],
                a_recons_points[k][:, 1],
                a_recons_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_a_choice}_recons",
                axis = fig.add_subplot(10, 6, k*6+2, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_fake_points[k][:, 0],
                b_fake_points[k][:, 1],
                b_fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_a_choice}2{args.class_b_choice}",
                axis = fig.add_subplot(10, 6, k*6+3, projection='3d')
            )
            _ = plot_3d_point_cloud(
                a_fake_points[k][:, 0],
                a_fake_points[k][:, 1],
                a_fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_b_choice}2{args.class_a_choice}",
                axis = fig.add_subplot(10, 6, k*6+4, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_recons_points[k][:, 0],
                b_recons_points[k][:, 1],
                b_recons_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_b_choice}_recons",
                axis = fig.add_subplot(10, 6, k*6+5, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_points[k][:, 0],
                b_points[k][:, 1],
                b_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_b_choice}_real",
                axis = fig.add_subplot(10, 6, k*6+6, projection='3d')
            )
        fig.savefig(
            os.path.join(
                args.save_img_path, f"{epoch}.png"
            )
        )
        plt.close(fig)
        break

def validate_MVS(a_test_dataloader, b_test_dataloader_, encoder, generator, args, epoch, all_a_mean, all_b_mean):
    b_test_dataloader = iter(b_test_dataloader_)
    for _iter, data in enumerate(a_test_dataloader, 0):
        encoder.eval()
        generator.eval()
        
        a_points, _ = data
        a_points = a_points.cuda()
        
        b_points, _ = b_test_dataloader.next()
        b_points = b_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            a_z = encoder(a_points)
            b_z = encoder(b_points)
            b2a_z = b_z - all_b_mean + all_a_mean
            a2b_z = a_z - all_a_mean + all_b_mean
            b_fake_points = generator(a2b_z)
            a_fake_points = generator(b2a_z)
            a_recons_points = generator(a_z)
            b_recons_points = generator(b_z)

       
        a_points = a_points.detach().cpu().numpy()
        b_points = b_points.detach().cpu().numpy()
        a_fake_points = a_fake_points.detach().cpu().numpy()
        b_fake_points = b_fake_points.detach().cpu().numpy()
        a_recons_points = a_recons_points.detach().cpu().numpy()
        b_recons_points = b_recons_points.detach().cpu().numpy()

        
        fig = plt.figure(figsize=(30, 40))
        for k in range(args.batch_size):
            _ = plot_3d_point_cloud(
                a_points[k][:, 0],
                a_points[k][:, 1],
                a_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_a_choice}_real",
                axis=fig.add_subplot(10, 6, k*6+1, projection='3d')
            )
            _ = plot_3d_point_cloud(
                a_recons_points[k][:, 0],
                a_recons_points[k][:, 1],
                a_recons_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_a_choice}_recons",
                axis = fig.add_subplot(10, 6, k*6+2, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_fake_points[k][:, 0],
                b_fake_points[k][:, 1],
                b_fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_a_choice}2{args.class_b_choice}",
                axis = fig.add_subplot(10, 6, k*6+3, projection='3d')
            )
            _ = plot_3d_point_cloud(
                a_fake_points[k][:, 0],
                a_fake_points[k][:, 1],
                a_fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_b_choice}2{args.class_a_choice}",
                axis = fig.add_subplot(10, 6, k*6+4, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_recons_points[k][:, 0],
                b_recons_points[k][:, 1],
                b_recons_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_b_choice}_recons",
                axis = fig.add_subplot(10, 6, k*6+5, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_points[k][:, 0],
                b_points[k][:, 1],
                b_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{epoch}_{k}_{args.class_b_choice}_real",
                axis = fig.add_subplot(10, 6, k*6+6, projection='3d')
            )
        fig.savefig(
            os.path.join(
                args.save_img_path, f"{epoch}.png"
            )
        )
        plt.close(fig)
        break

def save_all_images(a_test_dataloader, b_test_dataloader_, encoder, generator, a2b_transfer, b2a_transfer, args):
    for _iter, data in enumerate(a_test_dataloader, 0):
        encoder.eval()
        generator.eval()
        a2b_transfer.eval()
        
        a_points, a_name = data
        a_points = a_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():

            a_z = encoder(a_points)
            a2b_z = a2b_transfer(a_z)
            b_fake_points = generator(a2b_z)
            a_recons_points = generator(a_z)
        
        a_points = a_points.detach().cpu().numpy()
        b_fake_points = b_fake_points.detach().cpu().numpy()
        a_recons_points = a_recons_points.detach().cpu().numpy()

        
        for k in range(a_points.shape[0]):
            fig = plt.figure(figsize=(5, 15))
            _ = plot_3d_point_cloud(
                a_points[k][:, 0],
                a_points[k][:, 1],
                a_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{k}_{args.class_a_choice}_real",
                axis = fig.add_subplot(311, projection='3d')
            )
            _ = plot_3d_point_cloud(
                a_recons_points[k][:, 0],
                a_recons_points[k][:, 1],
                a_recons_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{k}_{args.class_a_choice}_recons",
                axis = fig.add_subplot(312, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_fake_points[k][:, 0],
                b_fake_points[k][:, 1],
                b_fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{k}_{args.class_a_choice}2{args.class_a_choice}",
                axis = fig.add_subplot(313, projection='3d')
            )
            fig.savefig(
                os.path.join(
                    args.save_a_img_path, f"{_iter}_{k}_{a_name[k].split('.')[0]}.png"
                )
            )
            plt.close(fig)

    for _iter, data in enumerate(b_test_dataloader_, 0):
        encoder.eval()
        generator.eval()
        b2a_transfer.eval()
        
        b_points, b_name = data
        b_points = b_points.cuda()
        
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():

            b_z = encoder(b_points)
            b2a_z = b2a_transfer(b_z)
            a_fake_points = generator(b2a_z)
            b_recons_points = generator(b_z)

        
        b_points = b_points.detach().cpu().numpy()
        a_fake_points = a_fake_points.detach().cpu().numpy()
        b_recons_points = b_recons_points.detach().cpu().numpy()

        
        for k in range(b_points.shape[0]):
            fig = plt.figure(figsize=(5, 15))
            _ = plot_3d_point_cloud(
                b_points[k][:, 0],
                b_points[k][:, 1],
                b_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{k}_{args.classb_choice}_real",
                axis = fig.add_subplot(311, projection='3d')
            )
            _ = plot_3d_point_cloud(
                b_recons_points[k][:, 0],
                b_recons_points[k][:, 1],
                b_recons_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{k}_{args.class_b_choice}_recons",
                axis = fig.add_subplot(312, projection='3d')
            )
            _ = plot_3d_point_cloud(
                a_fake_points[k][:, 0],
                a_fake_points[k][:, 1],
                a_fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title=f"{k}_{args.class_b_choice}2{args.class_a_choice}",
                axis = fig.add_subplot(313, projection='3d')
            )
            fig.savefig(
                os.path.join(
                    args.save_b_img_path, f"{_iter}_{k}_{b_name[k].split('.')[0]}.png"
                )
            )
            plt.close(fig)
 

def save_shape_style_mixing_images(test_dataloader, encoder, generator, path, args):
    
    for _iter, data in enumerate(test_dataloader, 0):
        encoder.eval()
        generator.eval()
        points, name = data
        points = points.cuda()

        with torch.no_grad():
            z = encoder(points)
            source_z = z[:args.batch_size//2]
            target_z = z[args.batch_size//2:]
            
            z_1 = source_z.clone()
            z_1[:,128:] = target_z[:,128:].clone()
            
            z_2 = target_z.clone()
            z_2[:,128:] = source_z[:,128:].clone()
            
            fake_points = generator(z)
            fake_points_1 = generator(z_1)
            fake_points_2 = generator(z_2)


        points = points.detach().cpu().numpy()
        fake_points = fake_points.detach().cpu().numpy()
        fake_points_1 = fake_points_1.detach().cpu().numpy()
        fake_points_2 = fake_points_2.detach().cpu().numpy()

        
        for k in range(args.batch_size//2):
            fig = plt.figure(figsize=(30, 5))
            _ = plot_3d_point_cloud(
                points[k][:, 0],
                points[k][:, 1],
                points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title="source",
                axis = fig.add_subplot(161, projection='3d')
            )
            _ = plot_3d_point_cloud(
                fake_points[k][:, 0],
                fake_points[k][:, 1],
                fake_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                title="recons",
                axis = fig.add_subplot(162, projection='3d')
            )
            _ = plot_3d_point_cloud(
                fake_points_1[k][:, 0],
                fake_points_1[k][:, 1],
                fake_points_1[k][:, 2],
                in_u_sphere=True,
                show=False,
                title="source_z(:128)",
                axis = fig.add_subplot(163, projection='3d')
            )
            _ = plot_3d_point_cloud(
                fake_points_2[k][:, 0],
                fake_points_2[k][:, 1],
                fake_points_2[k][:, 2],
                in_u_sphere=True,
                show=False,
                title="target_z(:128)",
                axis = fig.add_subplot(164, projection='3d')
            )  
            _ = plot_3d_point_cloud(
                fake_points[args.batch_size//2 + k][:, 0],
                fake_points[args.batch_size//2 + k][:, 1],
                fake_points[args.batch_size//2 + k][:, 2],
                in_u_sphere=True,
                show=False,
                title="recons",
                axis = fig.add_subplot(165, projection='3d')
            )
            _ = plot_3d_point_cloud(
                points[args.batch_size//2 + k][:, 0],
                points[args.batch_size//2 + k][:, 1],
                points[args.batch_size//2 + k][:, 2],
                in_u_sphere=True,
                show=False,
                title="target",
                axis = fig.add_subplot(166, projection='3d')
            )
            fig.savefig(
                os.path.join(
                    path, f"{_iter}_{k}_{name[k].split('.')[0]}.png"
                )
            )
            plt.close(fig)

        print(_iter) 


def plot_tsne(a_test_dataLoader, b_test_dataLoader, encoder, a2b_transfer, b2a_transfer, generator, args):
    all_real_a_z = []
    all_fake_b_z = []
    all_a_points = []
    all_a_recons_points = []
    for _iter, data in enumerate(a_test_dataLoader, 0):
        encoder.eval()
        a2b_transfer.eval()
        b2a_transfer.eval()
        generator.eval()
        
        a_points, _ = data
        a_points = a_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            a_z = encoder(a_points)
            a2b_z = a2b_transfer(a_z)
            a_recons_points = generator(a_z)
            b_fake_points = generator(a2b_z)
            fake_a2b_z = encoder(b_fake_points)

        all_real_a_z.append(a_z.detach().cpu().numpy())
        all_fake_b_z.append(a2b_z.detach().cpu().numpy())
        all_a_points.append(a_points.detach().cpu().numpy())
        all_a_recons_points.append(a_recons_points.detach().cpu().numpy())
        
    all_real_a_z = torch.tensor(all_real_a_z).cuda().view(-1, 256)
    all_fake_b_z = torch.tensor(all_fake_b_z).cuda().view(-1, 256)



    all_real_b_z = []
    all_fake_a_z = []
    all_b_points = []
    all_b_recons_points = []
    for _iter, data in enumerate(b_test_dataLoader, 0):
        encoder.eval()
        a2b_transfer.eval()
        b2a_transfer.eval()
        generator.eval()
        
        b_points, _ = data
        b_points = b_points.cuda()
        
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            b_z = encoder(b_points)
            b2a_z = b2a_transfer(b_z)
            b_recons_points = generator(b_z)
            a_fake_points = generator(b2a_z)
            fake_b2a_z = encoder(a_fake_points)

        all_real_b_z.append(b_z.detach().cpu().numpy())
        all_fake_a_z.append(b2a_z.detach().cpu().numpy())
        all_b_points.append(b_points.detach().cpu().numpy())
        all_b_recons_points.append(b_recons_points.detach().cpu().numpy())

    all_fake_a_z = torch.tensor(all_fake_a_z).cuda().view(-1, 256)
    all_real_b_z = torch.tensor(all_real_b_z).cuda().view(-1, 256)


    all_real_a_z -= all_real_a_z.mean(0)
    all_real_b_z -= all_real_b_z.mean(0)
    all_fake_a_z -= all_fake_a_z.mean(0)
    all_fake_b_z -= all_fake_b_z.mean(0)


    all_z = torch.cat([all_real_a_z, all_real_b_z, all_fake_a_z, all_fake_b_z], dim=0)


    print("tsne-----------")
    all_z_tsne = manifold.TSNE(n_components=2).fit_transform(all_z)
    #Data Visualization
    x_min, x_max = all_z_tsne.min(0), all_z_tsne.max(0)
    X_norm = (all_z_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(8, 8))
    
    p1_num = all_real_a_z.size(0)
    p2_num = all_real_a_z.size(0) + all_real_b_z.size(0)
    p3_num = all_real_a_z.size(0) + all_real_b_z.size(0) + all_fake_a_z.size(0)
 
    p1 = plt.scatter(X_norm[:p1_num, 0], X_norm[:p1_num, 1], marker = 'o', color='r', s=10) #real chair
    p2 = plt.scatter(X_norm[p1_num:p2_num, 0], X_norm[p1_num:p2_num, 1], marker = 'o', color='g', s=10)  #real table
    p3 = plt.scatter(X_norm[p2_num:p3_num, 0], X_norm[p2_num:p3_num, 1], marker = 'o', color='b', s=10)  #fake chair
    p4 = plt.scatter(X_norm[p3_num:, 0], X_norm[p3_num:, 1], marker = 'o', color='y', s=10) #fake table

    plt.legend([p1, p2, p3, p4], ['input chair', 'input table', 'transferred chair', 'transferred table'], loc='lower right', scatterpoints=1)

    plt.xticks([])
    plt.yticks([])
    plt.show()


def save_MVS_images(a_test_dataLoader, b_test_dataLoader, encoder, generator, args):
    all_a_z = []
    all_b_z = []
    for _iter, data in enumerate(a_test_dataLoader, 0):
        encoder.eval()
        a_points, _ = data
        a_points = a_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            a_z = encoder(a_points)

        all_a_z.append(a_z.detach().cpu().numpy())
    
    for _iter, data in enumerate(b_test_dataLoader, 0):
        encoder.eval()
        b_points, _ = data
        b_points = b_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            b_z = encoder(b_points)

        all_b_z.append(b_z.detach().cpu().numpy())

    all_a_z = torch.tensor(all_a_z).cuda().view(-1, 256)
    all_b_z = torch.tensor(all_b_z).cuda().view(-1, 256)
    all_a_mean = all_a_z.mean(0)
    all_b_mean = all_b_z.mean(0)


    for _iter, data in enumerate(a_test_dataLoader, 0):
        encoder.eval()
        a_points, a_name = data
        a_points = a_points.cuda()
        with torch.no_grad():
            a_z = encoder(a_points)     
            a_z -= all_a_mean
            a_z += all_b_mean
            fake_a_points = generator(a_z)
        
        a_points = a_points.detach().cpu().numpy()
        fake_a_points = fake_a_points.detach().cpu().numpy()

        for k in range(a_points.shape[0]):
            fig = plt.figure(figsize=(5, 10))
            _ = plot_3d_point_cloud(
                a_points[k][:, 0],
                a_points[k][:, 1],
                a_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                axis = fig.add_subplot(211, projection='3d')
            )
            _ = plot_3d_point_cloud(
                fake_a_points[k][:, 0],
                fake_a_points[k][:, 1],
                fake_a_points[k][:, 2],
                in_u_sphere=True,
                show=False,
                axis = fig.add_subplot(212, projection='3d')
            )
            fig.savefig(
                os.path.join(
                    args.save_img_path, f"{_iter}_{k}_{a_name[k].split('.')[0]}.png"
                )
            )
            plt.close(fig)        


def save_MVS_npy(a_test_dataLoader, b_test_dataLoader, encoder, generator, args):
    all_a_points = []
    all_b_points = []
    all_a_z = []
    all_b_z = []
    for _iter, data in enumerate(a_test_dataLoader, 0):
        encoder.eval()
        a_points, _ = data
        a_points = a_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            a_z = encoder(a_points)

        all_a_z.append(a_z.detach().cpu().numpy())
        all_a_points.append(a_points.detach().cpu().numpy())
    
    for _iter, data in enumerate(b_test_dataLoader, 0):
        encoder.eval()
        b_points, _ = data
        b_points = b_points.cuda()
        
        # ---------------------- Encoder ---------------------- #
        with torch.no_grad():
            b_z = encoder(b_points)

        all_b_z.append(b_z.detach().cpu().numpy())
        all_b_points.append(b_points.detach().cpu().numpy())

    all_a_z = torch.tensor(all_a_z).cuda().view(-1, 256)
    all_b_z = torch.tensor(all_b_z).cuda().view(-1, 256)
    all_a_points = torch.tensor(all_a_points).cuda().view(-1, 2048, 3)
    all_b_points = torch.tensor(all_b_points).cuda().view(-1, 2048, 3)
    all_a_mean = all_a_z.mean(0)
    all_b_mean = all_b_z.mean(0)



    for ps in args.points_index:
        with torch.no_grad():
            
            a_z = all_a_z[ps].view(-1, 256)     
            a_z -= all_a_mean
            a_z += all_b_mean
            fake_a_points = generator(a_z)
        
        fake_a_points = fake_a_points.detach().cpu().numpy()

        all_points = np.concatenate((all_a_points[ps], fake_a_points[0], fake_a_points[0]), axis=0)
      
        np.save(f"{args.plot_npy_path}/short_tall_{ps}.npy", all_points)


def save_shape_style_mixing_npy(test_dataset, encoder, generator, args):
    
    a_points, _ = test_dataset[args.points_index_1]
    b_points, _ = test_dataset[args.points_index_2]
    
    encoder.eval()
    generator.eval()
        
    a_points = torch.tensor(a_points).cuda().unsqueeze(0)
    b_points = torch.tensor(b_points).cuda().unsqueeze(0)
    # ---------------------- Encoder ---------------------- #
    with torch.no_grad():
        a_z = encoder(a_points)
        b_z = encoder(b_points)
        ab_z = a_z.clone()
        ab_z[:,128:] = b_z[:,128:].clone()
        ab_points = generator(ab_z)    
    a_points = a_points.detach().cpu().numpy()
    b_points = b_points.detach().cpu().numpy()
    ab_points = ab_points.detach().cpu().numpy()
    all_points = np.concatenate((a_points[0], ab_points[0], b_points[0]), axis=0)
        
    np.save(f"{args.plot_npy_path}/{args.points_index_1}_{args.points_index_2}.npy", all_points)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config", default="configs/config_chair_table.yaml", type=str
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config["common"].items():
        setattr(args, k, v)
    for k, v in config["test"].items():
        setattr(args, k, v)

    test(args)