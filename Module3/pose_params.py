


import yaml
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

""" 
    Loading the pose parameters 
"""


class SensorCalibration:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        elif self.config_path.endswith('.json'):
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = json.load(file)
        else:
            raise ValueError("unsupport type.")
    
    def get_tactile_pose(self):
        tactile = self.config['sensors']['tactile']
        return {
            'translation': np.array(tactile['translation']),
            'rotation': np.array(tactile['rotation']),
            'rotation_matrix': self.euler_to_matrix(tactile['rotation']),
            'clip_th': float(tactile['clip_th'])
        }
    
    def get_visual_pose(self):
        visual = self.config['sensors']['visual']
        return {
            'translation': np.array(visual['translation']),
            'rotation': np.array(visual['rotation']),
            'rotation_matrix': self.euler_to_matrix(visual['rotation'])
        }
    
    def euler_to_matrix(self, euler_angles, degrees=True):
        return R.from_euler('xyz', euler_angles, degrees=degrees).as_matrix()
    
    def get_metadata(self):
        return self.config.get('metadata', {})
    
    def print_calibration_info(self):
        tactile = self.get_tactile_pose()
        visual = self.get_visual_pose()
        metadata = self.get_metadata()
        



        