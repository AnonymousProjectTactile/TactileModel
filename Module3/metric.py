


import numpy as np 

from scipy.spatial import Delaunay 
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors 

   
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
    

def metric_cal(X, Y_):
    error_cham = chamfer_distance(X, Y_)
    error_hau = hausdorff_distance(X, Y_)
    return error_cham, error_hau


def compute_knn_indices(X, k=6):
    tree = cKDTree(X)
    _, indices = tree.query(X, k=k+1)
    return indices[:, 1:]  



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
            

