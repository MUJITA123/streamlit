import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology
from scipy import ndimage

def siv_function_all(src_dir, output_xlsx_path):
    # Get all image paths
    img_paths = []
    for ext in ['*.png', '*.bmp', '*.PNG', '*.BMP']:
        img_paths.extend(glob.glob(os.path.join(src_dir, ext)))

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
        'Compactness': []
    }

    for img_path in img_paths:
        img = cv2.imread(img_path)
        print(f"Processing image: {img_path}")
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Process the image to obtain vessel information
        processed_img, vessel_img, region_area, vessel_area, density, loop_num = siv_hole_area(img)

        # Extract features from the image
        bud_num = siv_budding_num_2(img)
        perimeter, aspect_ratio, solidity, rectangularity, compactness = siv_other(img, processed_img)

        # Store the results
        data['Sample name'].append(os.path.basename(img_path))
        data['Loop number'].append(loop_num)
        data['Region area'].append(region_area)
        data['Vessel density'].append(density)
        data['Vessel area'].append(vessel_area)
        data['Leading bud number'].append(bud_num)
        data['Perimeter'].append(perimeter)
        data['Aspect ratio'].append(aspect_ratio)
        data['Solidity'].append(solidity)
        data['Rectangularity'].append(rectangularity)
        data['Compactness'].append(compactness)

    # Save results to Excel
    df = pd.DataFrame(data)
    df.to_excel(output_xlsx_path, index=False)

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

src_dir = "./SIV"
output_xlsx_path = 'SIVP.xlsx'
start_time = time.time()
siv_function_all(src_dir, output_xlsx_path)
print(f"Time taken: {time.time() - start_time:.2f} seconds")
