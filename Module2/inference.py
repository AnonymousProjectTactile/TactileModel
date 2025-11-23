import sys
import torch
import cv2
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
import os
sys.path.append("Module2/")
import scipy
from utils1.data_loading import BasicDataset

print(scipy.__version__)
from model.unet import UNet  # segmentation
from model.RL import *  # policy

MEMORY_CAPACITY = 1000
REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
][0]  
def mask_to_image(mask: np.ndarray):  
        if mask.ndim == 2:
            return Image.fromarray((mask * 255).astype(np.uint8))
        elif mask.ndim == 3:
            return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

class Explor_Infer:

    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = args
        self.segment_net = UNet(n_channels=3, n_classes=2, bilinear=False).to(device=self.device)
        self.policy = DDPG(state_dim=2,
                           action_dim=1,
                           action_bound=1,
                           replacement=REPLACEMENT,
                           memory_capacity=MEMORY_CAPACITY)
        self._load_model()
        self.segment_net.eval()

    def _load_model(self):
        """ line detection + edge following """
        self.segment_net.load_state_dict(torch.load(self.config.model_unet_path))
        self.policy.load(self.config.policy_path)

    def pre_process(self, frame):
        target_size = 128
        width, height = 640, 480
        center_x, center_y = width // 2, height // 2
        left = center_x - target_size // 2
        top = center_y - target_size // 2
        right = center_x + target_size // 2
        bottom = center_y + target_size // 2
        crop_img = frame[top:bottom, left:right]
        return crop_img
    
    def segment(self, img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = torch.from_numpy(BasicDataset.preprocess(img_pil, 0.5, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = self.segment_net(img)
            probs = torch.sigmoid(output)[0]
            tf = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((img_pil.size[1], img_pil.size[0])),
                                     transforms.ToTensor()])
            full_mask = tf(probs.cpu()).squeeze()
        mask = (full_mask > 0.5).numpy()
        mask = mask[0, :, :]
        mask = 1 - mask  
        result = mask_to_image(mask)
        result = np.array(result)
        result = np.expand_dims(result, axis=-1) 
        return result

    def inference(self, img):
        crop_img = self.pre_process(img) 
        img_seg = self.segment(crop_img)
        s = obtain_edge3_1(img_seg)  #
        a = self.policy.choose_action(s)
        a = np.clip(a, -1, 1)
        a = a * np.pi * 0.5
        x, y = 0.01 * np.cos(a), 0.01 * np.sin(a)
        x, y = float(x),float(y)
        x, y= str(round(x * 200, 1)),str(round(-y * 200, 1))

        return x,y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_unet_path', type=str, default='')
    parser.add_argument('--policy_path', type=str, default='')
    parser.add_argument('--img_path', type=str, default='')
    args = parser.parse_args()

    Exploration = Explor_Infer(args)
    img_path = args.img_path
    img = cv2.imread(img_path)
    action_x,action_y = Exploration.inference(img)
    print(action_x,action_y)


