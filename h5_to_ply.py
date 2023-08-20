#!/usr/local/bin/ipython --gui=wx
# Adapted from SPLOCS view_animation.py by Thomas Neumann, see https://github.com/tneumann/splocs
# Copyright (c) [2015] [Javier Romero]

from argparse import ArgumentParser
import h5py
from itertools import count
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input
import pyvista
import numpy as np
import open3d as o3d
import random

def normalize_batch(points):
    bb_max = points.max(0)
    bb_min = points.min(0)
    length = (bb_max - bb_min).max()
    
    mean = (bb_max + bb_min) / 2.0
    points = (points - mean) /length
    return points


def save_ply_file(all_verts, tris, key_names, file_name):
    for _iter, verts in enumerate(all_verts):
        verts = normalize_batch(verts)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.compute_vertex_normals()

        pcd = mesh.sample_points_uniformly(number_of_points=2048)
        o3d.io.write_point_cloud(f"{file_name}/{_iter}_{key_names[_iter]}.ply", pcd)

def save_imgs(all_verts, tris, key_names, file_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512)
    
    for _iter, verts in enumerate(all_verts):
        verts = normalize_batch(verts)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.compute_vertex_normals()

        vis.add_geometry(mesh)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{file_name}/{_iter}_{key_names[_iter]}.jpg")
        vis.clear_geometries()
    vis.destroy_window()


def main(hdf5_animation_file, sid='50004'):
    train_verts = []
    valid_verts = []
    test_verts = []
    
    train_names = []
    valid_names = []
    test_names = []

    with h5py.File(hdf5_animation_file, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            if sid in key:
                verts = f[key].value.transpose([2,0,1])
                indices = [i for i in range(verts.shape[0])]
                random.shuffle(indices)
                verts = verts[indices]
                train_verts.append(verts[:int(0.7*len(verts))])
                valid_verts.append(verts[int(0.7*len(verts)):int(0.8*len(verts))])
                test_verts.append(verts[int(0.8*len(verts)):])

                train_names = train_names + [key] * int(0.7*len(verts))
                valid_names = valid_names + [key] * (int(0.8*len(verts)) - int(0.7*len(verts)))
                test_names = test_names + [key] * (len(verts) - int(0.8*len(verts)))
                
        tris = f['faces'].value

    train_verts = np.concatenate(train_verts, axis=0)
    valid_verts = np.concatenate(valid_verts, axis=0)
    test_verts = np.concatenate(test_verts, axis=0)



    print("save training data")
    save_ply_file(train_verts, tris, train_names, "../data/fit-fat/fat_train")
    print("save valid data")
    save_ply_file(valid_verts, tris, valid_names, "../data/fit-fat/fat_valid")
    print("save testing data")
    save_ply_file(test_verts, tris, test_names, "../data/fit-fat/fat_test")


    # print("save training data")
    # save_imgs(train_verts, tris, train_names, "../data/fit-fat/fit_train_plot")
    # print("save valid data")
    # save_imgs(valid_verts, tris, valid_names, "../data/fit-fat/fat_valid_plot")
    # print("save testing data")
    # save_imgs(test_verts, tris, test_names, "../data/fit-fat/fat_test_plot")

if __name__ == '__main__':

    sids = ['50004', '50020', '50021', '50022', '50025',
            '50002', '50007', '50009', '50026', '50027']


    parser = ArgumentParser(description='Save sequence meshes as obj')
    parser.add_argument('--path', type=str, default='./dyna_female.hdf5',
                        help='dataset path in hdf5 format')
    parser.add_argument('--sid', type=str, default='50004',
                        choices=sids, help='subject id')
    args = parser.parse_args()

    main(args.path, args.sid)
