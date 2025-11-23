

import sys 
sys.path.append('Module3')
import os
import time
import argparse
import matplotlib
# matplotlib.use('Qt5Agg')
import numpy as np
import numpy.matlib
import scipy.sparse
import scipy.io
import open3d as o3d
from functools import partial
import matplotlib.pyplot as plt
from regist_em import * 
from metric import *  
from pose_params import * 
from utils_pc import *


""" basic functions of struction """


def preprocess_visual(visual_all, init_visual_trans, init_visual_rot):
    """ ====== Visual Process 
        Load
        Transformation 
        ======  
    """ 
    visual_all = extract_outer_surface( visual_all , grid_size=0.5)
    len_visual = len(visual_all)
    # print("length of visual pc:", len_visual)
    # o3d_show_numpy(visual_all)

    pcd_visual = o3d.geometry.PointCloud()
    pcd_visual.points = o3d.utility.Vector3dVector(visual_all)
    mesh_visual = reconstruct_alpha_shape(pcd_visual, show=False)
    # mesh_visual = reconstruct_Possion(pcd_visual)
    mesh_sample_visual = mesh_visual.sample_points_poisson_disk(number_of_points=len_visual*5)
    mesh_sample_visual = np.asarray(mesh_sample_visual.points)
    mesh_sample_visual = extract_outer_surface( mesh_sample_visual , grid_size=0.5)
    # print_pc_info(mesh_sample_visual,print_flag=True)
    # o3d_show_numpy(mesh_sample_visual)
    # global_map_new = mesh_sample_visual

    transform_visual = create_transform_matrix(init_visual_trans, init_visual_rot)
    visual_homogeneous = np.column_stack([mesh_sample_visual, np.ones(len(mesh_sample_visual))])
    transformed_visual = (transform_visual @ visual_homogeneous.T).T[:, 0:3]
    mesh_sample_visual = transformed_visual
    return mesh_sample_visual



def preprocess_tactile(local_pc_0, Tx, Ty, Tz, init_sensor_rot, tactile_clip):
    """ ====== Tactile Process 
        Load 
        Transformation 
        ====== 
    """
    flatten_pc_0 = flatten_pc(local_pc_0, pixmm=(0.05, 0.05, 1))  
    # o3d_show_numpy(flatten_pc_0)
    # print_pc_info(flatten_pc_0, print_flag=True)        
    flatten_pc_0 = clip_pc(flatten_pc_0, max_z=tactile_clip) 
    transform = create_transform_matrix([Tx, Ty, Tz], init_sensor_rot)

    flatten_pc_0_homogeneous = np.column_stack([flatten_pc_0, np.ones(len(flatten_pc_0))])
    transformed_pc = (transform @ flatten_pc_0_homogeneous.T).T[:, 0:3]
    return transformed_pc



def extract_patch(global_pc, local_pc):
    """ extract corresponding local patch from global view """
    min_x, max_x, min_y, max_y, min_z, max_z = print_pc_info(local_pc, print_flag=False) 
    min_bound = [min_x, min_y, -np.inf]
    max_bound = [max_x, max_y, np.inf]
    local_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    global_patch = extract_points_in_bbox(global_pc, local_bbox) 
    return global_patch 



def reconstruction(visual, tactile, method, lam=10):
    """ registration """
    n_samples = 2 
    for i in range(n_samples):
        indices = np.random.choice(len(tactile), len(visual), replace=False)
        tactile_sampled = tactile[indices]
        reg = EM(**{'alpha': lam , 'beta': 2, 'X': visual,'Y': tactile_sampled })
        reg.register()
        Y_ = reg.TY.detach().cpu().numpy() 
        if i ==0:
            local_map_refine = Y_
        local_map_refine = merge_maps(local_map_refine, Y_) # local sample combination 
    return local_map_refine 





