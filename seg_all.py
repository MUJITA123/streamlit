import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon
def seg_function_all(img, img_name="default_image"):
    # img_paths = []
    # for ext in ['*.png', '*.bmp']:  # Reduce to two extensions to avoid redundancy
    #     img_paths.extend(glob.glob(os.path.join(src_dir, ext)))
    # img_paths = list(set(img_paths))  # Remove duplicates using set

    # Initialize data storage
    data = {
        'Sample name': [],
        'Loop number': [],
        'Region area': [],
        'Vessel density': [],
        'Vessel area': [],
        'Leading bud number': [],
        'Perimeter': [],
        'Aspect ratio': [],
        'Solidity': [],
        'Rectangularity': [],
        'Compactness': [],
        'Diameter':[],
        'Irregularity':[],
        'Vessel_length':[],
        'Total_interval_length':[]
    }

    # for img_path in img_paths:
    #     img = cv2.imread(img_path)
    #     print(f"Processing image: {img_path}")
    #     if img is None:
    #         print(f"Failed to load image: {img_path}")
    #         continue

        # Process the image to obtain vessel information
    processed_img, vessel_img, region_area, vessel_area, density, loop_num = siv_hole_area(img)

    # Extract features from the image
    bud_num = siv_budding_num_2(img)

    perimeter, aspect_ratio, solidity, rectangularity, compactness = siv_other(img, processed_img)

    Diameter_value=Diameter(img)

    Irregularity_value=Irregularity(img)

    vessel_length, Total_interval_length=ISV(img)

    # Store the results
    # data['Sample name'].append(os.path.basename(img))#img_path
    data['Sample name'].append(img_name)
    #data['Loop number'].append(loop_num)
    data['Region area'].append(region_area)
    data['Vessel density'].append(density)
    data['Vessel area'].append(vessel_area)
    #data['Leading bud number'].append(bud_num)
    data['Perimeter'].append(perimeter)
    #data['Aspect ratio'].append(aspect_ratio)
    data['Solidity'].append(solidity)
    data['Rectangularity'].append(rectangularity)
    #data['Compactness'].append(compactness)
    data['Irregularity'].append(Irregularity_value)
    data['Diameter'].append(Diameter_value)
    data['Vessel_length'].append(vessel_length)
    #data['Total_interval_length'].append(Total_interval_length)

# Save results to Excel
    df = pd.DataFrame(data)
    return df

def siv_hole_area(img_input):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Morphological operations (Erosion, Dilation, and Hole Filling)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel)
    dilated = cv2.dilate(eroded, kernel)

    # Remove small objects and fill holes
    cleaned = morphology.remove_small_objects(dilated.astype(bool), min_size=200)
    filled = ndimage.binary_fill_holes(cleaned)

    # Convert to float for further processing
    filled_img = filled.astype(np.uint8) * 255

    # Apply the image threshold for vessel detection (Otsu-like thresholding)
    f = img_input.astype(np.float64) / 255
    T = 0.4 * (np.min(f) + np.max(f))  # Otsu's threshold approximation
    done = False
    while not done:
        g = f >= T
        Tn = 0.4 * (np.mean(f[g]) + np.mean(f[~g]))
        done = abs(T - Tn) < 0.1
        T = Tn

    # Binary vessel image
    vv = (f >= T).astype(np.uint8)

    # Apply morphological operations to isolate vessel structures
    vessel_img = morphology.remove_small_objects(vv.astype(bool), min_size=500)
    vessel_img = ndimage.binary_fill_holes(vessel_img)

    # Label regions and compute the vessel area
    labels = measure.label(vessel_img, connectivity=2)
    regions = measure.regionprops(labels)
    vessel_area = sum([r.area for r in regions])

    # Total region area (for full connected structure)
    region_labels = measure.label(filled.astype(bool), connectivity=2)
    region_props = measure.regionprops(region_labels)
    region_area = sum([r.area for r in region_props])

    # Vessel density (ratio of vessel area to region area)
    density = vessel_area / region_area if region_area != 0 else 0

    # Euler number (loop number) - number of connected components
    loop_num = len(regions)

    return filled_img, vessel_img, region_area, vessel_area, density, loop_num

def siv_budding_num_2(img):
    # Placeholder function for budding number calculation
    return 0  # Actual logic for budding number goes here

def siv_other(img, processed_img):
    # Perimeter calculation
    processed_img_bool = processed_img.astype(bool)
    perimeter = measure.perimeter(processed_img_bool)

    # Minimum bounding rectangle
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, 0, 0, 0, 0

    rect = cv2.minAreaRect(contours[0])
    width, height = rect[1]
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

    # Solidity: Area of the region / Area of the convex hull
    convex_img = cv2.convexHull(contours[0])
    solidity = cv2.contourArea(contours[0]) / cv2.contourArea(convex_img)
    # Rectangularity: Area of bounding box / Area of region
    rectangularity = (width * height) / measure.regionprops(processed_img)[0].area if width > 0 and height > 0 else 0
    # Compactness: (Perimeter^2) / (4 * pi * Area)
    compactness = (perimeter ** 2) / (4 * np.pi * measure.regionprops(processed_img)[0].area) if perimeter > 0 else 0
    return perimeter, aspect_ratio, solidity, rectangularity, compactness

def Diameter(image):
        # 1. 二值化
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.where(image > 0, 1, 0).astype(np.uint8)
    # 2. 连通区域分析
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)
    total_area = sum([r.area for r in regions]) if regions else 0
    # 3. 骨架化
    skeleton = morphology.skeletonize(binary)
    # 4. 移除小对象（可选，尝试调整 min_size 或跳过此步骤）
    min_size = 2  # 减小 min_size（原为 256），根据图像调整
    cleaned = morphology.remove_small_objects(skeleton.astype(bool), min_size=min_size)
    total_length = np.count_nonzero(cleaned)
    # 5. 计算直径
    if total_length == 0:
        print("Warning: total_length is zero after cleaning.")
        return 0.0
    diameter = total_area / total_length
    return diameter

def minboundrect(x, y, metric='a'):
    x, y = np.array(x), np.array(y)
    if len(x) != len(y) or len(x) < 1:
        return [], [], 0, 0

    if len(x) <= 2:
        if len(x) == 1:
            return [x[0]]*5, [y[0]]*5, 0, 0
        dx, dy = x[1] - x[0], y[1] - y[0]
        perimeter = 2 * np.sqrt(dx**2 + dy**2)
        return [x[0], x[1], x[1], x[0], x[0]], [y[0], y[1], y[1], y[0], y[0]], 0, perimeter

    # 凸包计算
    points = np.vstack((x, y)).T
    hull = ConvexHull(points)
    x, y = points[hull.vertices, 0], points[hull.vertices, 1]
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # 旋转角度
    edgeangles = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    edgeangles = np.unique(np.mod(edgeangles, np.pi/2))

    area, perimeter = float('inf'), float('inf')
    best_rectx, best_recty = [], []
    for theta in edgeangles:
        rot = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])
        xyr = points @ rot
        xymin, xymax = np.min(xyr, axis=0), np.max(xyr, axis=0)
        A_i = np.prod(xymax - xymin)
        P_i = 2 * np.sum(xymax - xymin)
        M_i = A_i if metric == 'a' else P_i

        if M_i < (area if metric == 'a' else perimeter):
            area = A_i
            perimeter = P_i
            rect = np.array([xymin, [xymax[0], xymin[1]], xymax, [xymin[0], xymax[1]], xymin])
            rect = rect @ rot.T
            best_rectx, best_recty = rect[:, 0], rect[:, 1]

    return best_rectx, best_recty, area, perimeter
def Irregularity(image):
    # file_list = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.bmp'))]
    # results = []
    # for filename in file_list:
    #     img_path = os.path.join(src_dir, filename)
    #     image = cv2.imread(img_path)
            # 1. 转换为灰度图并二值化
    J = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    J = np.where(J > 0, 255, 0).astype(np.uint8)
    # 2. 形态学处理
    se = morphology.disk(3)
    J = morphology.erosion(J, se)  # 腐蚀
    J = morphology.dilation(J, se)  # 膨胀
    J = morphology.remove_small_objects(J.astype(bool), min_size=1000, connectivity=2)  # 移除小区域
    J = measure.label(J) > 0  # 近似 imfill，填充孔洞
    J = J.astype(bool)  # 修正为布尔类型，确保后续操作一致
    # 3. 计算特征
    labeled = measure.label(J)
    regions = measure.regionprops(labeled)
    if not regions:
        return [0, 0, 0, 0, 0, 0]
    # Vessel area 和 Perimeter
    S = sum(r.area for r in regions)  # 总面积
    P = sum(r.perimeter for r in regions)  # 总周长

    # Aspect ratio 和 Rectangularity
    r, c = np.where(J == 1)
    rectx, recty, _, _ = minboundrect(c, r, 'p')  # 注意：c 是 x，r 是 y
    dd = np.array([rectx[:-1], recty[:-1]]).T
    dd1 = dd[[3, 0, 1, 2], :]
    ds = np.sqrt(np.sum((dd - dd1) ** 2, axis=1))
    kuan, chang = min(ds[:2]), max(ds[:2])
    ck = chang / kuan if kuan > 0 else 0  # Aspect ratio
    Sq = kuan * chang
    Spercent = S / Sq if Sq > 0 else 0  # Rectangularity
    # 重心计算
    height, width = J.shape
    sum_x, sum_y, area = 0, 0, 0
    for i in range(height):
        for j in range(width):
            if labeled[i, j] == 1:  # 只取第一个连通区域
                sum_x += i
                sum_y += j
                area += 1
    plot_x = int(sum_x / area) if area > 0 else 0
    plot_y = int(sum_y / area) if area > 0 else 0

    # Irregularity
    dilated = morphology.binary_dilation(J, morphology.disk(1))  # 膨胀一步
    contour = dilated & ~J  # 边界提取：膨胀后的区域与原区域的差集
    pos1 = np.array(np.where(contour)).T  # [y, x]
    d2 = (pos1[:, 0] - plot_x) ** 2 + (pos1[:, 1] - plot_y) ** 2
    Ir = np.pi * np.max(d2) / S if S > 0 else 0
    return Ir

def ISV_one_image(image):
    # 1. 二值化（模仿 MATLAB J 的处理）
    binary = np.where(image > 0, 255, 0).astype(np.uint8) / 255.0  # 转为 [0, 1]

    # 形态学处理
    se = morphology.disk(1)
    binary = morphology.erosion(binary, se)  # 腐蚀
    binary = morphology.dilation(binary, se)  # 膨胀
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=200, connectivity=2)  # 移除小区域
    binary = measure.label(binary) > 0  # 转为二值图并填充孔洞（近似 imfill）

    # 2. 计算 Aspect ratio 和 Rectangularity
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    if not regions:
        return [0, 0, 0, 0]

    largest_region = max(regions, key=lambda r: r.area)
    coords = largest_region.coords
    rect = cv2.minAreaRect(coords)  # 最小外接矩形
    width, height = rect[1]
    kuan, chang = min(width, height), max(width, height)
    aspect_ratio = chang / kuan if kuan > 0 else 0
    Sq = kuan * chang
    St = sum(r.area for r in regions)
    rectangularity = St / Sq if Sq > 0 else 0

    # 3. 计算 Vessel length 和 Total interval length
    # ROI 和 BW1 的近似处理
    _, BW = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BW = BW / 255.0
    BW = morphology.erosion(BW, se)
    BW = morphology.dilation(BW, se)
    BW = morphology.remove_small_objects(BW.astype(bool), min_size=100, connectivity=2)
    ROI = binary
    pos1 = np.array(np.where(ROI > 0)).T  # [y, x] 坐标
    if len(pos1) < 2:
        return [aspect_ratio, rectangularity, 0, 0]
    # 找到左右端点
    d1 = pos1[:, 0] ** 2 + pos1[:, 1] ** 2
    d2 = pos1[:, 0] ** 2 + (pos1[:, 1] - ROI.shape[1]) ** 2
    ind1, ind2 = np.argmin(d1), np.argmin(d2)
    x1, y1 = pos1[ind1, 1], pos1[ind1, 0]
    x2, y2 = pos1[ind2, 1], pos1[ind2, 0]
    # 沿路径采样 75 个点
    x11 = np.linspace(x1, x2, 75, dtype=int)
    y11 = []
    for x in x11:
        col = ROI[:, x]
        y_idx = np.where(col > 0)[0]
        y11.append(y_idx[0] if len(y_idx) > 0 else y1)  # 简化处理
    y11 = np.array(y11)
    # 计算路径长度 d（Total interval length）
    d = 0
    for i in range(len(x11) - 1):
        d += np.sqrt((y11[i] - y11[i + 1]) ** 2 + (x11[i] - x11[i + 1]) ** 2)
    vessel_length = St / d if d > 0 else 0
    return [vessel_length, d]#aspect_ratio, rectangularity,
def ISV(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vessel_length, Total_interval_length= ISV_one_image(image)
    return vessel_length, Total_interval_length

# print(seg_function_all(r"C:\Users\mujita\Documents\MATLAB\seg\ISV").to_string())
