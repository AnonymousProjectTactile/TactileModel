
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import math 
import random 
from scipy import stats
import statistics

def calculate_intersection(line1, line2):
    """
    计算两条直线的交点
    line1 和 line2: 直线信息列表，每个元素为 [t
    返回交点 (x, y) 坐标
    """
    a1, b1 = line1[1], line1[2]  # 直线1的斜率和截距
    a2, b2 = line2[1], line2[2]  # 直线2的斜率和截距
    if a1 == a2:
        return None  # 两条直线平行，没有交点
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (x, y)

def calculate_angle(x1, y1, x2, y2):
    """
    计算点 (x1, y1) 和 (x2, y2) 之间的角度，返回 [0, 360] 范围内的角度
    """
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if angle < 0:
        angle += 360
    return angle

def get_quadrant(angle):
    """
    根据角度判断所在的象限
    """
    if 0 <= angle < 90:
        return "第一象限"
    elif 90 <= angle < 180:
        return "第二象限"
    elif 180 <= angle < 270:
        return "第三象限"
    else:
        return "第四象限"
    

def point_to_line_distance(k, intercept, point=(32, 32)):
    """
    Func: 计算 【点point】到【以倾斜角度和截距所确定的直线】的距离
    - angle: 倾斜角度 ,水平朝右为0,顺时针旋转为0-360度
    - intercept: 截距
    Out:  输出距离 
    """
    a = k  # 斜率
    b = intercept  # 截距
    A = -a
    B = 1
    C = -b
    x1, y1 = point
    distance = abs(A * x1 + B * y1 + C) / np.sqrt(A**2 + B**2)
    y_line = a * x1 + b
    if y1 > y_line:
        position = 1.  # 点在直线上方
    elif y1 < y_line:
        position = -1. # 下方
    else:
        position = 0.  # 点在直线上
    return distance*position


def filter_lines_duplicate(lines ):
    """ 
    Func: 去除斜率相同的重复的或错误的line 
    Input:
        - lines: 待筛选的直线的集合
    Output: 
        - filtered_lines: 筛选之后的直线 
    """
    threshold=5
    filtered_lines = []
    for line in lines:
        angle1, slope1, intercept1, x1, y1, x2, y2 = line
        is_unique = True
        for filtered_line in filtered_lines:
            angle2, slope2, intercept2, x3, y3, x4, y4 = filtered_line
            angle_diff = abs(angle1 - angle2)
            if angle_diff <= threshold:
                dist1 = point_to_line_distance(slope1, intercept1, (32, 32))
                dist2 = point_to_line_distance(slope1, intercept2, (32, 32))
                if dist1 < dist2:
                    filtered_lines.remove(filtered_line)  # 删除远的那条
                    filtered_lines.append(line)  # 保留当前直线
                is_unique = False
                break
        if is_unique:
            filtered_lines.append(line)
    return filtered_lines

def filter_lines_orth(lines):
    """ 
    Func: 找正交的直线 
    Input:
        - lines: 待筛选的直线的集合
    Output: 
        - filtered_lines: 筛选之后的直线 
    """
    angle_threshold = 16
    perpendicular_lines = []
    for i, (angle1, a1, intercept1, x11, y11, x21, y21) in enumerate(lines):
        for j, (angle2, a2, intercept2, x12, y12, x22, y22) in enumerate(lines[i + 1:], start=i + 1):
            angle_diff = abs(angle1 - angle2) % 360  # 计算角度差，并确保是 0 到 360 度范围内
            if abs(angle_diff - 90) <= angle_threshold or abs(angle_diff - 270) <= angle_threshold:
                perpendicular_lines.append([angle1, a1, intercept1, x11, y11, x21, y21])
                perpendicular_lines.append( [angle2, a2, intercept2, x12, y12, x22, y22])
    if not perpendicular_lines:
        closest_line = None
        min_distance = float('inf')  # 初始化一个非常大的距离
        for (angle1, a1, intercept1, x11, y11, x21, y21) in lines:
            distance = point_to_line_distance(a1, intercept1, (32, 32))
            if distance < min_distance:
                min_distance = distance
                closest_line = [[angle1, a1, intercept1, x11, y11, x21, y21]]
        return closest_line
    return perpendicular_lines
        

def sample_points( contour):
    sampled_points = []
    y_indices, x_indices = np.where(contour == 255)
    for _ in range(10):
        random_idx = random.choice(range(len(x_indices)))  # 从所有的索引中随机选择一个
        random_x = x_indices[random_idx]
        random_y = y_indices[random_idx]
        sampled_points.append((random_x, random_y))
    return sampled_points



    
def select_closest_line(lines, theta3):

    def angle_distance(theta1, theta2):
        return min(abs(theta1-theta2), abs((theta1+180)%360-theta2))
    
    closest_line = min(lines, key=lambda line: angle_distance(line[0], theta3))
    return closest_line
    

def calculate_angle_from_origin(x, y):
    """ 
    Func: 计算从原点到点的角度，返回范围是 [0, 360) 度
    """
    angle = math.atan2(y, x)  # atan2返回的结果是 [-π, π] 范围，单位为弧度
    angle_deg = math.degrees(angle)  # 转换为度
    if angle_deg < 0:
        angle_deg += 360  # 保证返回值在 [0, 360) 范围内
    return angle_deg



def get_region_from_sample(intersection, point, directions):
    # 0,  90,  180, 270
    x_diff = point[0] - intersection[0]
    y_diff = point[1] - intersection[1]
    angle = calculate_angle_from_origin(x_diff, y_diff)
    if angle >= directions[0] and angle < directions[1]:
        # print("Region 1")
        return directions[1]
    elif angle >= directions[1] and angle < directions[2]:
        # print("Region 2")
        return directions[2]
    elif angle >= directions[2] and angle < directions[3]:
        # print("Region 3")
        return directions[3]
    else:
        # print("Region 4")
        return directions[0]
    

    
def obtain_sign(k,b, points):
    signs = []
    # k, b = line[0], line[1], line[2]
    signs = [(y - (k * x + b)) > 0 for x, y in points]
    sign = statistics.mode(signs) 
    return float(sign) 
    



def obtain_edge(depth_image):
    """ 
    Func: 
    Input:
        - depth_image: 输入深度图
    Output: 
        - 返回待跟踪的直线信息 
    """
    blurred_image = cv2.GaussianBlur(depth_image, (5, 5), 0) # blur 
    _, binary_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY) # binary 
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contour 
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < 250:  # 判断最大轮廓的面积是否小于250
            return None  # 返回 None
        result_image = np.zeros_like(binary_image)
        cv2.drawContours(result_image, [max_contour], -1, (255), thickness=cv2.FILLED)
    for contour in contours:
        x,y,bw,bh = cv2.boundingRect(contour)
        if x == 0 or y==0 or (x+bw)>=64 or (y+bw)>= 64:
            edge_flag =  1
        else:
            edge_flag = 0
        if not edge_flag:
            return None 
    edges = cv2.Canny(result_image, 50, 150, apertureSize=3) # edge 
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=6, minLineLength=6, maxLineGap=20) # all_lines 
    if lines is None: # 检测不到线 
        return np.array([0,0,0])
    # print("Detected lines: ", len(lines))
    image_with_lines = np.zeros_like(binary_image) 
    lines_info = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # 线段的两个端点坐标
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (255), 2)
            if x2 != x1: 
                a = (y2 - y1) / (x2 - x1)
            else:
                a = 1000 
            b = y1 - a * x1
            theta_rad = math.atan(a)
            theta = math.degrees(theta_rad)
            if theta < 0:
                theta += 180
            # print(f"直线的角度 θ: {theta} 度, 直线的截距 b: {b}, 直线的斜率{a}.")
            lines_info.append([theta, a, b, x1, y1, x2, y2])
    
    ## filter duplicate line 
    filter_lines = filter_lines_duplicate(lines_info) # deplicate 
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(result_image) # 联通域 
    plt.title('Contour')
    plt.subplot(1, 3, 2)
    plt.imshow(edges) # all_lines
    plt.title('All_lines')
    plt.subplot(1, 3, 3)
    plt.imshow(image_with_lines) # filtered_lines   
    plt.title('Filtered_lines')
    plt.show()
        
    # print("Filtered duplicate lines (len): ", len(filter_lines))
    if len(filter_lines) == 0:
        return None
    elif len(filter_lines) == 1:
        points = sample_points( result_image)
        sign = obtain_sign(filter_lines[0], points)
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(result_image) # 联通域 
        # plt.title('Contour')
        # plt.subplot(1, 3, 2)
        # plt.imshow(edges) # all_lines
        # plt.title('All_lines')
        # plt.subplot(1, 3, 3)
        # plt.imshow(image_with_lines) # filtered_lines   
        # plt.title('Filtered_lines')
        # plt.show()
        1
        return np.array([filter_lines[0][1],filter_lines[0][2], sign]) # angle,b,sign
    else: 
        filter_lines = filter_lines_orth(filter_lines) # orth 
        # print("Filtered orth lines (len): ", len(filter_lines))
        if len(filter_lines) == 1:
            points = sample_points( result_image)
            sign = obtain_sign(filter_lines[0], points)
            return np.array([filter_lines[0][1],filter_lines[0][2], sign])
        if len(filter_lines) == 0 :
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(result_image) # 联通域 
                plt.title('Contour')
                plt.subplot(1, 3, 2)
                plt.imshow(edges) # all_lines
                plt.title('All_lines')
                plt.subplot(1, 3, 3)
                plt.imshow(image_with_lines) # filtered_lines   
                plt.title('Filtered_lines')
                plt.show()
        # print(filter_lines)
        k1_line1 = filter_lines[0][0]
        k2_line1 = filter_lines[0][0] + 180 
        k1_line2 = filter_lines[1][0]
        k2_line2 = filter_lines[1][0] + 180 
        directions = np.sort( np.array( [k1_line1, k2_line1, k1_line2, k2_line2]))
        intersection = calculate_intersection(filter_lines[0], filter_lines[1])
        # print(intersection)
        points = sample_points( result_image)
        angles = []
        for point in points:
            angle = get_region_from_sample(intersection, point, directions)
            angles.append(angle)
        angle = statistics.mode(angles)
        # print(directions)
        # print("angle: ", angle)
        line = select_closest_line(filter_lines, angle)
        sign = obtain_sign(line, points)
        # print(sign)
        lines_info
        res = np.array([line[1], line[2], sign ])
        # print(res)
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(result_image) # 联通域 
        # plt.title('Contour')
        # plt.subplot(1, 3, 2)
        # plt.imshow(image_with_lines) # all_lines
        # plt.title('All_lines')
        # plt.subplot(1, 3, 3)
        # plt.imshow(image_with_lines) # filtered_lines   
        # plt.title('Filtered_lines')
        # plt.show()
        return res 
        
    
    1
    
    
def obtain_edge1(depth_image, theta):
    """ 
    Func: 
    Input:
        - depth_image: 输入深度图
    Output: 
        - 返回待跟踪的直线信息 
    """
    depth_image = depth_image[10:-10, 10:-10]
    blurred_image = cv2.GaussianBlur(depth_image, (5, 5), 0) # blur 
    _, binary_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY) # binary 
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contour 
    
        
    if not contours:
        return None
    else:
        max_contour = max(contours, key=cv2.contourArea)
        result_image = np.zeros_like(binary_image)
        cv2.drawContours(result_image, [max_contour], -1, (255), thickness=cv2.FILLED)
    
    points = sample_points( result_image)
    edges = cv2.Canny(result_image, 50, 150, apertureSize=3) # edge 
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=6, minLineLength=6, maxLineGap=20) # all_lines 
    if lines is None: # 检测不到线 
        return None 
        return np.array([10,10,10,10,10])
    # print("Detected lines: ", len(lines))
    image_with_lines = np.zeros_like(binary_image) 
    lines_info = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # 线段的两个端点坐标
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (255), 2)
            dx = x2 - x1
            dy = y2 - y1
            theta1 = math.atan2(dy, dx)
            if x2 != x1: 
                a = (y2 - y1) / (x2 - x1)
            else:
                a = 1000 
            b = y1 - a * x1
            distance = point_to_line_distance(a,b)
            sign = obtain_sign(a,b, points)
            lines_info.append([theta1, a,b, distance, sign])
    lines_info
    
    theta_img = - theta + np.pi/2
    if theta_img > (np.pi/2):
        theta_img -= np.pi 
    thresh = 0.15
    filtered_lines = []
    for line in lines_info:
        if line[0] > (theta_img-thresh) and line[0] < (theta_img+thresh):
            filtered_lines.append(line)
    if len(filtered_lines) == 0:
        # print(1)
        # plt.figure()
        # plt.subplot(1, 4, 1)
        # plt.imshow(depth_image) # 联通域 
        # plt.title('Depth')
        # plt.subplot(1, 4, 2)
        # plt.imshow(result_image) # 联通域 
        # plt.title('Contour')
        # plt.subplot(1, 4, 3)
        # plt.imshow(edges) # all_lines
        # plt.title('All_lines')
        # plt.subplot(1, 4, 4)
        # plt.imshow(image_with_lines) # filtered_lines   
        # plt.title('Filtered_lines')
        # plt.show()
        # print(1)
        return None
    else:
        filterli = [filtered_lines[0][0], filtered_lines[0][3]*0.1 ,filtered_lines[0][4]]
        return filterli
    
    
def obtain_edge2(img, angle):
    ## env angle ---> standard angle
    angle = -angle + 90
    if angle > 180:
        angle = angle % (-360)
            
    center = np.array([64,64])
    ret, binary_image = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    edge_points = largest_contour.reshape(-1, 2) 
    distances = np.linalg.norm(edge_points - center, axis=1)
    nearest_point = edge_points[np.argmin(distances)]
    if nearest_point[0] < 8 or nearest_point[0] > 120 or nearest_point[1] < 8 or nearest_point[1] > 120:
        return None
        
    normal_vector = nearest_point - center
    normal_angle = np.arctan2(normal_vector[1], normal_vector[0])
    tangent_angle_clockwise = normal_angle + np.pi / 2
    tangent_angle_counterclockwise = normal_angle - np.pi / 2
    
    if tangent_angle_clockwise > np.pi:
        tangent_angle_clockwise = tangent_angle_clockwise % (-np.pi)
    if tangent_angle_counterclockwise < -np.pi:
        tangent_angle_counterclockwise = tangent_angle_counterclockwise % (np.pi)
    tangent_angle_clockwise =np.degrees(tangent_angle_clockwise)
    tangent_angle_counterclockwise =np.degrees(tangent_angle_counterclockwise)
    diff_clockwise = abs(tangent_angle_clockwise - angle)
    diff_counterclockwise = abs(tangent_angle_counterclockwise - angle)
    if diff_clockwise < diff_counterclockwise:
        angle1 =  tangent_angle_clockwise
    else:
        angle1 =   tangent_angle_counterclockwise

    distance, sign = point_to_line(angle1, nearest_point[0], nearest_point[1])
    dis = (distance*sign)/64.
    
    radian = np.deg2rad(angle)
    # 计算cos和sin
    cos_val = np.cos(radian)
    sin_val = np.sin(radian)
    
    angle1 = angle1/180. 
    
    return angle1, dis
    
    

def point_to_line(theta, x1,y1,x0=64, y0=64):
    """
    计算图像中心点O(64,64) 到直线之间的位置关系。
        theta 和 (x1,y1) 就确定了这条直线了
        然后计算(x0,y0)和直线之间的关系 
    """
    theta_rad = np.radians(theta)
    if abs(theta) == 90:
        distance = abs(x0-x1)
        if x1 > x0:
            pos = 1
        elif x1 < x0:
            pos = -1 
        else:
            pos = 0
    else:
        m = np.tan(theta_rad)
        A,B = m, -1
        C = y1 - m * x1 
        distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        y0_pred = m * x0 + C  # 直线上对应 xO 的 y 值
        if y0_pred > y0:
            pos = 1
        elif y0_pred < y0:
            pos = -1
        else:
            pos = 0
    return distance, pos


            
def obtain_edge3(img):
        ret, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        
        center = np.array([64,64])
        ret, binary_image = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        edge_points = largest_contour.reshape(-1, 2) 
        distances = np.linalg.norm(edge_points - center, axis=1)
        nearest_point = edge_points[np.argmin(distances)]
        if nearest_point[0] < 8 or nearest_point[0] > 120 or nearest_point[1] < 8 or nearest_point[1] > 120:
            return None
        normal_vector = nearest_point - center # 法线
        normal_angle = np.arctan2(normal_vector[1], normal_vector[0])
        normal_angle = np.degrees(normal_angle)
        
        tangent_angle_clockwise = normal_angle + 90
        tangent_angle_counterclockwise = normal_angle - 90
        if abs(tangent_angle_clockwise) < 90:
            angle = tangent_angle_clockwise
        elif abs(tangent_angle_counterclockwise) < 90:
            angle = tangent_angle_counterclockwise
        else:
            angle = -90 
        angle ### 
        if angle != 90:
            if nearest_point[1] > 64 :
                dis_sign = 1
            elif nearest_point[1] < 64 :
                dis_sign = -1
            else: 
                dis_sign = 0
        else: 
            if nearest_point[0] > 64 :
                dis_sign = 1
            elif nearest_point[0] < 64 :
                dis_sign = -1
            else: 
                dis_sign = 0
        distance = np.linalg.norm(nearest_point - center)
        distance_with_sign = distance * dis_sign
        distance_with_sign ### 
        # print('angle: ', angle , '  distance: ', distance_with_sign)
        return angle, distance_with_sign
    
