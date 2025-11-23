 
import sys 
sys.path.append('Module3/')

import os 
import cv2 
import glob
import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay  
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors 
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# from tools import * 
from CPD.torchcpd.deformReg import * 


data_path = 'Module3/Data/CPD_data/'
   
def chamfer_distance(X,Y_):
    tree_Y = cKDTree(Y_)
    tree_X = cKDTree(X)
    dist_XY, _ = tree_Y.query(X)
    dist_YX,_ = tree_X.query(Y_)
    return np.mean(dist_XY**2), np.mean(dist_YX**2)


def hausdorff_distance(X,Y_):
    tree_Y = cKDTree(Y_)
    tree_X = cKDTree(X)
    dist_XY, _ = tree_Y.query(X)
    dist_YX,_ = tree_X.query(Y_)
    return max(np.max(dist_XY),  np.max(dist_YX))
    

def arap_loss(Y_, Y, k=6):
    N = Y.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Y) 
    _, indices = nbrs.kneighbors(Y)
    loss = 0.0 
    
    for i in range(N):
        yi = Y[i]
        yi_ = Y_[i]

        nei_idx = indices[i][1:]
        Yj = Y[nei_idx]        # (k, 3)
        Yj_ = Y_[nei_idx]  # (k, 3)

        v = Yj - yi          # (k, 3)
        v_ = Yj_ - yi_       # (k, 3)

        H = v.T @ v_
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:  # handle reflection case
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        diff = v_ - (v @ R.T)
        loss += np.sum(np.linalg.norm(diff, axis=1) ** 2)

    return loss / N


def metric_cal(X, Y_, Y=None):
    """ evaluate the performance of registration"""
    # error_nn_mse = nn_mse(X, Y_)
    if Y is None:
        Y = Y_
    error_cham = chamfer_distance(X, Y_)
    error_hau = hausdorff_distance(X, Y_)
    error_arap = arap_loss(Y_, Y )
    return error_cham, error_hau, error_arap


def vis_registration_error_comparision(target_PC, source_PC_1, source_PC_2):
    def create_standalone_colorbar(vmin=0.0, vmax=5.0):
        fig, ax = plt.subplots(figsize=(2, 6))
        fig.subplots_adjust(right=0.5)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=plt.cm.Reds, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label('Registration Error (mm)', fontsize=22)
        cbar.ax.tick_params(labelsize=24)
        plt.savefig('colorbar.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        print("Colorbar saved as 'colorbar.png'. Please include this image in your paper.")

    def compute_errors(source, target):
        nn =  NearestNeighbors(n_neighbors=1)
        nn.fit(target)
        distances, _ = nn.kneighbors(source)
        return distances.flatten()
    err1 = compute_errors(source_PC_1, target_PC)
    err2 = compute_errors(source_PC_2, target_PC)
    all_errors = np.concatenate([err1, err2])
    # min_err, max_err = all_errors.min(), all_errors.max()
    min_err, max_err = 0., 5.
    create_standalone_colorbar(vmin=min_err, vmax=max_err)
    def error_to_colors(errors):
        norm = (errors - min_err) / (max_err - min_err + 1e-8)
        # cmap = plt.get_cmap("jet")
        cmap = plt.get_cmap("Reds")
        return cmap(norm)[:, :3]  # RGB
    def make_colored_pcd(points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_PC)
    pcd_target.paint_uniform_color([1, 1, 1])  #
    
    pcd_method1 = make_colored_pcd(source_PC_1, error_to_colors(err1))
    pcd_method2 = make_colored_pcd(source_PC_2, error_to_colors(err2))
    
    print("Press 'n' to toggle between geometries")
    o3d.visualization.draw_geometries([ pcd_method1], window_name="ICP")
    o3d.visualization.draw_geometries([pcd_method2], window_name="CPD")
    
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = plt.Normalize(vmin=min_err, vmax=max_err)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'),
                       cax=ax, orientation='horizontal')
    cb1.set_label('Registration Error (Euclidean Distance)')
    plt.show()
    print(1)
    

def refine_map_icp(global_map, local_map):
    """ Refine the map """
    
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(local_map)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(global_map)
    
    threshold = 5  
    initial_transformation = np.eye(4, dtype=np.float64)
    initial_transformation[:3, 3] = np.array([0,0,0])
    result_icp = o3d.pipelines.registration.registration_icp(
                                            source_pcd, 
                                            target_pcd, 
                                            threshold, 
                                            initial_transformation,
                                            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source_pcd.transform(result_icp.transformation)
    transformed = np.asarray(source_pcd.points)
    return transformed 

  

def regist_single(data_path, k=0., b=0.):
    data = np.load(data_path, allow_pickle=True).item()
    global_pc = data['global_patch']
    local_pc = data['tactile_sample']  
    hardness = data['hardness'] 
    force = data['pressure'] 
    alpha = k * hardness / force + b
    # pc = np.concatenate((global_pc, local_pc), 0)
    # o3d_show_numpy(pc)
    # global_pc_show = o3d_show_numpy_set_color(global_pc, [1, 0, 0])  # 
    # local_pc_show = o3d_show_numpy_set_color(local_pc, [0, 0, 1])  # 
    # o3d.visualization.draw_geometries([global_pc_show, local_pc_show], window_name='CPD')
    reg = DeformableRegistration(**{'alpha': alpha ,  'beta': 2,  'X': global_pc ,   'Y': local_pc ,  'device': 'cuda:0'})
    reg.register()
    Y_ = reg.TY.detach().cpu().numpy() # transformed local_map 
    error_cham, error_hau, error_rarp = metric_cal(global_pc, Y_, local_pc )
    return error_cham[0], error_cham[1], error_hau, error_rarp
    
    
    
def WOA(objective_function, lb, ub, dim, num_whales = 20, max_iter = 20):
    a_decay = 2 / max_iter
    whales = np.random.uniform(lb, ub, (num_whales, dim)) 
    fitness = np.array([objective_function(whale) for whale in whales])
    best_idx = np.argmin(fitness) 
    best_whale = whales[best_idx].copy()
    best_score = fitness[best_idx] 
    convergence_curve = [] 
    trajectory_history = [] 
    print("Initial end ... ")
    
    for t in range(max_iter):
        print('Iter: ', t)
        a = 2 - a_decay * t 
        
        for i in range(num_whales):
            r = np.random.rand(dim) # 
            A = 2 * a * r - a # [-a, a]
            C = 2 * np.random.rand(dim) # random r2 
            p = np.random.rand()
            
            if p < 0.5: #
                if np.linalg.norm(A) < 1: 
                # exploitation 
                    D = np.abs(C * best_whale - whales[i]) 
                    whales[i] = best_whale - A * D 
                else:   
                # Exploration 
                    rand_whale = whales[np.random.randint(0, num_whales)]
                    D = np.abs(C * rand_whale - whales[i])
                    whales[i] = rand_whale - A * D
            else: 
            # 
                distance_to_leader = np.abs(best_whale - whales[i]) 
                b = 1 
                l = np.random.uniform(-1, 1)
                whales[i] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
            
            whales[i] = np.clip(whales[i], lb, ub)
            
        fitness = np.array([objective_function(w) for w in whales])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_score:
            best_score = fitness[min_idx]
            best_whale = whales[min_idx].copy()
        print("Best_whale: ", best_whale)
        convergence_curve.append(best_score)
        trajectory_history.append(best_whale.copy())
        
    return best_whale, best_score, convergence_curve, trajectory_history 
    
    

def objective_function(params):
    k,b = params
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = data_list[:100]
    errors_ij = []
    for data_idx in data_list: 
        error1, error2, error3, error4 = regist_single(data_path + data_idx, k=k, b=b)
        if error1 is None:
            continue
        error =  error1 + error2 + error3 + error4
        errors_ij.append(error)
    if len(errors_ij) == 0:
        return 1000
    error_ij = np.array(errors_ij).mean()
    return error_ij 
            

# lb = np.array([0.01, 0.01])
# ub = np.array([10, 1])
# dim = 2 
# best_params, best_value, convergence_curve, trajectory_history = WOA(objective_function, lb, ub, dim) 
# ## Vis - convergence_curve  
# plt.figure(figsize=(8, 4))
# plt.plot(convergence_curve, label='Best Score')
# plt.title('WOA Convergence Curve')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# print(1)
    





def test_Before():

    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = data_list[100:120]
    errors = []
    for file_path in data_list:
        # print(file_path)
        data = np.load(data_path + file_path, allow_pickle=True).item()
        global_pc = data['global_patch']
        local_pc = data['tactile_sample']  
        hardness = data['hardness'] 
        force = data['pressure'] 
        error_cham, error_hau, error_rarp = metric_cal(global_pc, local_pc, local_pc )
        errors.append([error_cham[0], error_cham[1], error_hau, error_rarp])
    mean_CPD = np.mean(np.array(errors), axis=0)   
    std_CPD = np.std(np.array(errors), axis=0)   
    print(mean_CPD)
    print(std_CPD)
    print(1)

def test_CPD_param(params):
    k,b = params
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = data_list[100:120]
    errors = []
    for file_path in data_list:
        # print(file_path)
        data = np.load(data_path + file_path, allow_pickle=True).item()
        global_pc = data['global_patch']
        local_pc = data['tactile_sample']  
        hardness = data['hardness'] 
        force = data['pressure'] 
        # X = global_patch
        alpha = k * hardness / force + b # 
        reg = DeformableRegistration(**{'alpha': alpha ,  'beta': 2,  'X': global_pc ,   'Y': local_pc ,  'device': 'cuda:0'})
        reg.register()
        Y_ = reg.TY.detach().cpu().numpy() # transformed local_map 
        Y_CPD = Y_
        error_cham, error_hau, error_rarp = metric_cal(global_pc, Y_, local_pc )
        errors.append([error_cham[0], error_cham[1], error_hau, error_rarp])
    mean_CPD = np.mean(np.array(errors), axis=0)   
    std_CPD = np.std(np.array(errors), axis=0)   
    print(mean_CPD)
    print(std_CPD)
    print(1)

def test_ICP():
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = data_list[100:120]
    errors = []
    for file_path in data_list:
        # print(file_path)
        data = np.load(data_path + file_path, allow_pickle=True).item()
        global_pc = data['global_patch']
        local_pc = data['tactile_sample']  
        hardness = data['hardness'] 
        force = data['pressure'] 
        Y_ = refine_map_icp(global_pc, local_pc)
        Y_ICP = Y_
        error_cham, error_hau, error_rarp = metric_cal(global_pc, Y_, local_pc )
        errors.append([error_cham[0], error_cham[1], error_hau, error_rarp])
    mean_CPD = np.mean(np.array(errors), axis=0)    # 沿第0轴（行）求平均，得到4个均值
    std_CPD = np.std(np.array(errors), axis=0)   
    print(mean_CPD)
    print(std_CPD)
    print(1)

def test_CPD_fix(lam=0.1):
    # 0.1 
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = data_list[100:120]
    errors = []
    for file_path in data_list:
        # print(file_path)
        data = np.load(data_path + file_path, allow_pickle=True).item()
        global_pc = data['global_patch']
        local_pc = data['tactile_sample']  
        hardness = data['hardness'] 
        force = data['pressure'] 
        # X = global_patch
        alpha = lam
        reg = DeformableRegistration(**{'alpha': alpha ,  'beta': 2,  'X': global_pc ,   'Y': local_pc ,  'device': 'cuda:0'})
        reg.register()
        Y_ = reg.TY.detach().cpu().numpy() # transformed local_map 
        Y_CPD = Y_
        error_cham, error_hau, error_rarp = metric_cal(global_pc, Y_, local_pc )
        errors.append([error_cham[0], error_cham[1], error_hau, error_rarp])
    mean_CPD = np.mean(np.array(errors), axis=0)   
    std_CPD = np.std(np.array(errors), axis=0)   
    print(mean_CPD)
    print(std_CPD)
    print(1)

def test():
    print(' ====== Before ====== ')
    test_Before()

    print(' ====== ICP ====== ')
    test_ICP()

    print(' ====== PG-CPD ====== ')
    test_CPD_param([3.044 ,0.390])

    print(' ====== CPD 0.1 ====== ')
    test_CPD_fix(0.1)

    print(' ====== CPD 1 ====== ')
    test_CPD_fix(1)

    print(' ====== CPD 10 ====== ')
    test_CPD_fix(10)

    print(' ====== CPD 20 ====== ')
    test_CPD_fix(20)


test()



