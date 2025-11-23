import numpy as np  
import torch
import torch.nn.functional as F
import torch.nn as nn

def Info_NCE_loss(z,z_pos, z_next):
	"""from paper: Learning Predictive Representations for Deformable Objects Using Contrastive Estimation."""
	bs = z_pos.shape[0]
	neg_dot_products = torch.mm(z_next, z.t()) # b x b # 矩阵乘     [128, 128]
	neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (z ** 2).sum(1).unsqueeze(0)) # (128, 128), why 是 （a-b）^2吗
	idxs = np.arange(bs)
	# Set to minus infinity entries when comparing z with z - will be zero when apply softmax
	neg_dists[idxs, idxs] = float('-inf') # b x b+1

	pos_dot_products = (z_pos * z_next).sum(dim=1) # b
	pos_dists = -((z_pos ** 2).sum(1) - 2* pos_dot_products + (z_next ** 2).sum(1)) # 128
	pos_dists = pos_dists.unsqueeze(1) # b x 1

	dists = torch.cat((neg_dists, pos_dists), dim=1) # b x b+1 # input:(128,128),(128,1) output:(128,129)
	dists = F.log_softmax(dists, dim=1) # b x b+1
	loss = -dists[:, -1].mean() # Get last column with is the true pos sample

	return loss

def L1_loss(pred, label):
	""" predict, target """
	loss_f = nn.L1Loss(reduction='mean')  #平均绝对误差,L1-损失
	loss = loss_f(pred,label)
	return loss

def L2_loss(pred, label):
	loss_f = nn.MSELoss()  #L2-损失
	loss = loss_f(pred,label)
	return loss

def BCE_loss(recon_img, img):
    return F.binary_cross_entropy(recon_img, img, reduction='sum')

def MSE_loss(recon_act, act):
	return F.mse_loss(recon_act.view(-1, 4), act.view(-1, 4), reduction='sum')

def loss_KL_constrain(d):
	b = d.shape[0]
	eps = torch.randn(( d.shape[1],d.shape[2],d.shape[3])).to('cuda')
	eps = eps.repeat(b, 1,1,1)
	kl = F.kl_div(d.softmax(dim=-1).log(), eps.softmax(dim=-1), reduction='mean')
	return kl

def loss_contrast( h1,h2):
	# contrast_loss
	margin = 2
	d = torch.mean(torch.abs(h1-h2))
	a = torch.tensor([margin-d, 0]).to('cuda')
	loss = torch.max(a)
	return loss



"""SSIM"""
import torch
import torch.nn.functional as F
from math import exp
import numpy as np

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
  
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

"""GAN loss"""
class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            target_tensor = target_tensor.expand_as(prediction)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss