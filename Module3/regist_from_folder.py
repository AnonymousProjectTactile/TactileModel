


import sys 
import os
import time
import argparse
from regist_VT import * 




def object_reconstruct(path, method):
    # ------ load config ------ 
    calib = SensorCalibration(path + '/pos.yaml')
    tactile_pose = calib.get_tactile_pose()
    visual_pose = calib.get_visual_pose()
    init_sensor_trans = tactile_pose['translation']  
    init_sensor_rot = tactile_pose['rotation']     
    tactile_clip =  tactile_pose['clip_th']  
    init_visual_trans = visual_pose['translation']   
    init_visual_rot = visual_pose['rotation']   
    visual_path = path + '/visual/visual.npy'
    tactile_path = path + '/tactile/'
    fusion_count = 0     
    refined_model = None 
    metric_cham0, metric_cham1, metric_hau = 0, 0, 0

    # ------ reconstruct ------ 
    visual_all = np.load(visual_path)
    mesh_visual = preprocess_visual(visual_all, init_visual_trans, init_visual_rot)
    local_maps = os.listdir(tactile_path)
    for local_name in local_maps:
        tactile_estim = np.load(tactile_path + local_name, allow_pickle=True).item()
        local_pc_0 = tactile_estim['geom']
        hardness = tactile_estim['hardness']
        force = tactile_estim['force']
        name = local_name.split(".")[0]
        Tx, Ty = float(name.split('_')[0]), float(name.split('_')[1])
        Tx = -Tx + init_sensor_trans[0] 
        Ty = -Ty + init_sensor_trans[1]
        Tz = init_sensor_trans[2]
        transformed_pc = preprocess_tactile(local_pc_0, Tx, Ty, Tz, init_sensor_rot, tactile_clip)
        if len(transformed_pc) < 50:
            continue
        visual_patch = extract_patch(mesh_visual, transformed_pc)
        if len(visual_patch) < 50:
            continue
        fusion_count += 1 
        k, b = 3.044 ,0.390
        lam = k * hardness / force + b # 
        local_map_refine = reconstruction(visual_patch, transformed_pc, method , lam=lam)
        if refined_model is None:
            refined_model = local_map_refine
        else:
            refined_model = np.concatenate([refined_model,local_map_refine], 0)
    

    
    return refined_model 






if __name__ == '__main__':


    obj = 'flower02_estim'   # 栀子花


    path = 'Module3/Data/SoftObj/flower/' + obj 
    method = 'PG_CPD' 
    model =  object_reconstruct(path, method)
    o3d_show_numpy(model)
    print('1')
    














