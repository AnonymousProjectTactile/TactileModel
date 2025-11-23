

import numpy as np 
import numpy as np
import numpy.matlib
import scipy.sparse
import scipy.io
import time
import open3d as o3d
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd


""" Tool Functions """

# =============================================================

# ====== Point Cloud Process ====== 

def print_pc_info(X, info='', print_flag = False):
    """ Print PC info """
    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
    min_z, max_z = np.min(X[:, 2]), np.max(X[:, 2])
    if print_flag:
        print(' ========= ')
        print(info + "_ X shape: ", X.shape)
        print("min_x: ", min_x, "max_x: ", max_x)
        print("min_y: ", min_y, "max_y: ", max_y)
        print("min_z: ", min_z, "max_z: ", max_z)
        print(" ========== ")
    return min_x, max_x, min_y, max_y, min_z, max_z


def o3d_show_numpy(X):
    """ PC visualization using open3d"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    min_height = np.min(X[:, 2])
    max_height = np.max(X[:, 2])
    normalized_heights = (X[:, 2] - min_height) / (max_height - min_height)
    cmap = plt.cm.get_cmap('rainbow')
    gradient_colors = cmap(normalized_heights)
    pcd.colors = o3d.utility.Vector3dVector(gradient_colors[:, :3])
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name="pc show", point_show_normal=False,width=800, height=600)
    # o3d.visualization.draw_geometries([pcd], window_name="pc show", point_show_normal=False,width=800, height=600)


def o3d_show_regis(X, Y):
    pcd_X = o3d.geometry.PointCloud()
    pcd_X.points = o3d.utility.Vector3dVector(X)
    pcd_X.paint_uniform_color([1,0,0])
    
    pcd_Y = o3d.geometry.PointCloud()
    pcd_Y.points = o3d.utility.Vector3dVector(Y)
    pcd_Y.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([pcd_X, pcd_Y], window_name="registration", point_show_normal=False,width=800, height=600)


def flatten_pc(pcd, pixmm=(0.072, 0.072, 100)):
    """ Faltten PC: Tactile image To Tactile PC """
    h, w = pcd.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    data = np.stack((x.flatten()*pixmm[0], \
                     y.flatten()*pixmm[1], \
                     -pcd.flatten()*pixmm[2]), axis=-1)
    return data


def clip_pc(X, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None):
    """ Clip PC """
    if min_x == None:
        min_x = np.min(X[:, 0])
    if max_x == None:
        max_x = np.max(X[:, 0])
    if min_y == None:    
        min_y = np.min(X[:, 1])
    if max_y == None:    
        max_y = np.max(X[:, 1])
    if min_z == None:
        min_z = np.min(X[:, 2])
    if max_z == None:
        max_z = np.max(X[:, 2])
    X = X[(X[:, 0] > min_x) & (X[:, 0] < max_x)]
    X = X[(X[:, 1] > min_y) & (X[:, 1] < max_y)]
    X = X[(X[:, 2] > min_z) & (X[:, 2] < max_z)]
    return X


def random_downsample(points, sample_ratio):
    """ Random Sample PC """
    num_samples = int(len(points) * sample_ratio)
    indices = np.random.choice(len(points), num_samples, replace=False)
    return points[indices]


def reconstruct_alpha_shape(pcd, show=False):
    """ Reconstruction form PC (Alpha_Shape) """
    alpha = 6
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    if show:
        o3d.visualization.draw_geometries([mesh], window_name="Alpha Shape",width=800,height=600, mesh_show_back_face=True)
    return mesh   


def extract_outer_surface(point_cloud, grid_size=0.2):
    """ Extract PC surface """
    # grid_size = 0.2 
    x,y,z = point_cloud[:,0], point_cloud[:,1], point_cloud[:,2]
    x_idx = np.floor(x / grid_size).astype(int)
    y_idx = np.floor(y / grid_size).astype(int)
    df = pd.DataFrame({
                    'x': x,
                    'y': y,
                    'z': z,
                    'x_idx': x_idx,
                    'y_idx': y_idx
                })
    top_df = df.loc[df.groupby(['x_idx', 'y_idx'])['z'].idxmin()]
    surface =  top_df[['x', 'y', 'z']].to_numpy()
    return surface 
    

def get_bounding_box(points, margin=5.0):
    """获取点云的边界框，并扩展边界"""
    min_bound = np.min(points, axis=0) - margin
    max_bound = np.max(points, axis=0) + margin
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def extract_points_in_bbox(points, bbox):
    """从点云中提取边界框内的点"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_in_bbox = pcd.crop(bbox)
    return np.asarray(pcd_in_bbox.points)


def create_transform_matrix(translation, rotation):
    """ Transformation Matrix """
    rotation = R.from_euler('xyz', rotation, degrees=True)
    Rot = rotation.as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = Rot
    T[0:3, 3] = translation
    return T


def merge_maps(global_map, refined_local_map):
    merged_map = np.vstack((global_map, refined_local_map)) 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_map)
    pcd = pcd.voxel_down_sample(0.05)
    return np.asarray(pcd.points)
 



