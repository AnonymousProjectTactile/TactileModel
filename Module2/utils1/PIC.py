from PIL import Image
import numpy as np

# 打开照片
image_path = r"C:\Users\23142\Desktop\Assignment8\dataset\train_data\masks\0cdf5b5d0ce1_01_mask.gif"
image = Image.open(image_path)

# 将照片转化成数组
image_array = np.array(image)

# 设置NumPy打印选项，限制输出的元素数量
np.set_printoptions(threshold=np.inf)

# 直接输出数组
print(image_array)
