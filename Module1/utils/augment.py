import numpy as np
import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image

#region RandAugment
"""code reference: https://github.com/taeoh-kim/temporal_data_augmentation
written by Taeoh Kim
"""
def temporal_interpolate(v_list, t, n):
    """
    v_list: [begin, end]
    t: i th
    n: num 
    """
    if len(v_list) == 1:
        return v_list[0]
    elif len(v_list) == 2:
        return v_list[0] + (v_list[1] - v_list[0]) * t / n
    else:
        NotImplementedError('Invalid degree')

class Augment:
    def __init__(self):
        pass

    def __call__(self, buffer):
        raise NotImplementedError

    def ShearX(self, imgs, v_list):  # [-0.3, 0.3]
        for v in v_list:
            assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        out = [img.transform(img.size, PIL.Image.AFFINE, (1, temporal_interpolate(v_list, t, len(imgs) - 1), 0, 0, 1, 0)) for t, img in enumerate(imgs)]
        return out

    def ShearY(self, imgs, v_list):  # [-0.3, 0.3]
        for v in v_list:
            assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        out = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, temporal_interpolate(v_list, t, len(imgs) - 1), 1, 0)) for t, img in enumerate(imgs)]
        return out

    def TranslateX(self, imgs, v_list):  # [-150, 150] => percentage: [-0.45, 0.45]
        for v in v_list:
            assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        v_list = [v * imgs.size[1] for v in v_list]
        out = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, temporal_interpolate(v_list, t, len(imgs) - 1), 0, 1, 0)) for t, img in enumerate(imgs)]
        return out

    def TranslateXabs(self, imgs, v_list):  # [-150, 150] => percentage: [-0.45, 0.45]
        for v in v_list:
            assert 0 <= v
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, temporal_interpolate(v_list, t, len(imgs) - 1), 0, 1, 0)) for t, img in enumerate(imgs)]
        return out

    def TranslateY(self, imgs, v_list):  # [-150, 150] => percentage: [-0.45, 0.45]
        for v in v_list:
            assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        v_list = [v * imgs.size[2] for v in v_list]

        out = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, temporal_interpolate(v_list, t, len(imgs) - 1))) for t, img in enumerate(imgs)]
        return out

    def TranslateYabs(self, imgs, v_list):  # [-150, 150] => percentage: [-0.45, 0.45]
        for v in v_list:
            assert 0 <= v
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, temporal_interpolate(v_list, t, len(imgs) - 1))) for t, img in enumerate(imgs)]
        return out

    def Rotate(self, imgs, v_list):  # [-30, 30]
        for v in v_list:
            assert -30 <= v <= 30
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        out = [img.rotate(temporal_interpolate(v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def AutoContrast(self, imgs, _):
        out = [PIL.ImageOps.autocontrast(img) for img in imgs]
        return out

    def Invert(self, imgs, _):
        out = [PIL.ImageOps.invert(img) for img in imgs]
        return out

    def Equalize(self, imgs, _):
        out = [PIL.ImageOps.equalize(img) for img in imgs]
        return out

    def Flip(self, imgs, _):  # not from the paper
        out = [PIL.ImageOps.mirror(img) for img in imgs]
        return out

    def Solarize(self, imgs, v_list):  # [0, 256]
        for v in v_list:
            assert 0 <= v <= 256

        out = [PIL.ImageOps.solarize(img, temporal_interpolate(v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Posterize(self, imgs, v_list):  # [4, 8]
        v_list = [max(1, int(v)) for v in v_list]
        v_list = [max(1, int(v)) for v in v_list]
        out = [PIL.ImageOps.posterize(img, int(temporal_interpolate(v_list, t, len(imgs) - 1))) for t, img in enumerate(imgs)]
        return out

    def Contrast(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Contrast(img).enhance(temporal_interpolate(v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Color(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Color(img).enhance(temporal_interpolate(v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Brightness(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Brightness(img).enhance(temporal_interpolate(v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Sharpness(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Sharpness(img).enhance(temporal_interpolate(v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Identity(self, imgs, _):
        return imgs

    def augment_list(self):
        # list of data augmentations and their ranges
        l = [(self.Identity, 0, 1), # original 无强度参数
            (self.AutoContrast, 0, 1),# 增强对比度 无强度参数
            (self.Equalize, 0, 1), # 直方图均衡化 无强度参数
            (self.Invert, 0, 1), # 反色
            (self.Rotate, 0, 30), # 旋转
            (self.Posterize, 0, 4), # 降低颜色位数
            (self.Solarize, 0, 256), # 反转超过阈值的像素质
            (self.Color, 0.1, 1.9),# 色彩平衡度
            (self.Contrast, 0.1, 1.9),# 对比度
            (self.Brightness, 0.1, 1.9),# 亮度
            (self.Sharpness, 0.1, 1.9),# 锐化
            (self.ShearX, 0., 0.3),
            (self.ShearY, 0., 0.3),
            (self.TranslateXabs, 0., 100),
            (self.TranslateYabs, 0., 100),]
        return l

class RandAugment(Augment):
    def __init__(self, n, m, temp_degree=1, range=1.0):
        super(RandAugment, self).__init__()
        self.max_severity = 30
        self.temp_degree = temp_degree
        self.n = n
        self.m = m  # usually values in the range [5, 30] is best
        self.range = range
        self.augment_list = self.augment_list()

    def __call__(self, buffer):
        buffer = [Image.fromarray(img.astype('uint8')) for img in buffer]
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if self.temp_degree == 0:
                val_list = [(float(self.m) / self.max_severity) * float(maxval - minval) + minval]
            else: # temp_degree == 1
                tval = float(np.random.uniform(low=0.0, high=0.5 * self.range * self.m)) # 0~2.5
                if random.random() > 0.5:
                    val_list = [((float(self.m) - tval) / self.max_severity) * float(maxval - minval) + minval]
                    val_list.extend([((float(self.m) + tval) / self.max_severity) * float(maxval - minval) + minval])
                else:
                    val_list = [((float(self.m) + tval) / self.max_severity) * float(maxval - minval) + minval]
                    val_list.extend([((float(self.m) - tval) / self.max_severity) * float(maxval - minval) + minval])
            buffer = op(buffer, val_list)
        buffer = np.array([np.array(img, np.dtype('float32')) for img in buffer])
        return buffer
# endregion  

#region SlowFast
"""
code reference: https://github.com/facebookresearch/SlowFast
# written by Hoseong Lee
"""

def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = images #torch.tensor(images)
    gray_channel = (
        0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray

def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `height` x `width` x`channel`
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `height` x `width` x`channel`
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    return images1 * alpha + images2 * (1 - alpha)

def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    images = images.astype(np.float32)
    return images

def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = np.zeros(images.shape) #torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images

def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    #img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    img_gray[:] = np.mean(img_gray, axis=(1, 2, 3),dtype=np.float32, keepdims=True)
    images = blend(images, img_gray, alpha)
    return images

def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images

def lighting_jitter(images, alphastd, eigval, eigvec):
    pass
#endregion 