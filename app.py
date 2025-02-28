import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
from PIL import Image
import glob
import io
import zipfile
from tools import *  # Assuming this includes get_largest_contour, make_mask_by_contours, extract_contour_rect
import base64
from seg_all import seg_function_all
import pandas as pd

# 设置页面配置（必须放在最前面）
st.set_page_config(page_title="Zebrafish AI", layout="wide")

# 自定义 CSS 设置背景颜色、导航文字样式和侧边栏收缩功能
st.markdown(
    """
    <style>
    .main {
        background-color: #FFFFF; /* 主页面白色 */
    }
    .stSidebar {
        background-color: #87CEEB; /* 导航栏蓝色 */
        transition: width 0.3s; /* 平滑过渡效果 */
    }
    .nav-link {
        color: #87CEEB; /* 导航文字浅蓝色 */
        font-size: 18px;
        text-decoration: none;
        margin: 10px 0;
        display: block;
        cursor: pointer;
    }
    .nav-link:hover {
        color: #FFFFFF; /* 鼠标悬停时变为白色 */
    }
    /* 收缩样式 */
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 0px !important;
        min-width: 0px !important;
    }
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 250px !important;
    }
    /* 右上角 Home 按钮样式 */
    .home-btn {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
    .home-btn button {
        font-size: 30px !important; /* 增大 Home 按钮字体大小 */
    }
    /* 自定义按钮样式 */
    div.stButton > button {
        background-color: white; /*##87CEEB*/
        color: black;
        padding: 15px 30px; /* 增大按钮内边距 */
        font-size: 24px; /* 增大“开始分析”按钮字体大小 */
        border: none !important; /* 强制移除边框 */
        border-radius: 5px;
        cursor: pointer;
        text-align: center; /* 确保按钮文字居中 */
    }
    div.stButton > button:hover {
        background-color: white;
    }
    /* 确保内容居中 */
    .centered-content {
        text-align: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 全局参数
DEVICE = "cpu"
SIZE = [416, 1024]
ENCODER = "se_resnext50_32x4d"
ENCODER_WEIGHTS = "imagenet"

# 模型权重选项
WEIGHTS_LIST = ["brain_area", "CCV", "CV2", "DA", "ISV", "PCV", "SIV"]
WEIGHTS_FILES = {
    "brain_area": "weights/brain_area_best_model_1024_416_cpu.pth",
    "CCV": "weights/CCV_best_model_1024_416_cpu.pth",
    "CV2": "weights/CV2_best_model_1024_416_cpu.pth",
    "DA": "weights/DA_best_model_1024_416_cpu.pth",
    "ISV": "weights/ISV_best_model_1024_416_cpu.pth",
    "PCV": "weights/PCV_best_model_1024_416_cpu.pth",
    "SIV": "weights/SIV_best_model_1024_416_cpu.pth"
}

# 缓存模型加载（10分钟过期）
@st.cache_resource(ttl=600)
def load_model(weight):
    file_path = WEIGHTS_FILES[weight]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模型文件 {file_path} 不存在，请检查 weights 文件夹。")
    model = torch.load(file_path, map_location=DEVICE)
    model.float()  # 将模型参数转换为 torch.float32
    model.eval()
    return model

# 图像预处理函数
def get_validation_augmentation(size):
    return lambda image: {"image": cv2.resize(image, (size[1], size[0]))}

def get_preprocessing(preprocessing_fn):
    return lambda image: {"image": preprocessing_fn(image)}

def preprocessing_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample = get_validation_augmentation(SIZE)(image=image)
    image = sample["image"]
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    sample = get_preprocessing(preprocessing_fn)(image=image)
    image = sample["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return image

# 分割图像并计算特征
def segment_image(_model, img, img_name="default_image"):
    with torch.no_grad():
        name = os.path.splitext(img_name)[0]  # Remove extension for cleaner filename
        img_org = img  # Input is already a NumPy array
        x_tensor = torch.from_numpy(preprocessing_img(img_org)).to(DEVICE).unsqueeze(0)
        pr_mask = _model(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask_org_size = cv2.resize(pr_mask, (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_CUBIC)
        pr_mask_org_size_threshold = pr_mask_org_size.round().astype(np.uint8) * 255

        # Draw bounding box
        contours, hierarchy = cv2.findContours(pr_mask_org_size_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = get_largest_contour(contours)
        mask = make_mask_by_contours(img_org.shape, img_org.dtype, [cnt])
        img_masked = np.where(mask, img_org, 0)
        mask_rect = extract_contour_rect(pr_mask_org_size_threshold, cnt)
        img_rect = extract_contour_rect(img_masked, cnt)
        print(f"Segmented image: {name}")
        # Calculate features using seg_function_all
        df_features = seg_function_all(img_masked, name)  # Pass the masked image and name directly

        return img_masked, df_features

# 创建 ZIP 文件并下载（处理 NumPy 数组）
def create_zip_of_results(images_to_process, selected_weights):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in images_to_process:
            image = np.array(Image.open(uploaded_file))
            image_name = uploaded_file.name if hasattr(uploaded_file, "name") else "demo_image"
            for weight in selected_weights:
                model = load_model(weight)
                segmented_image, _ = segment_image(model, image, image_name)
                img_buffer = io.BytesIO()
                Image.fromarray(segmented_image).save(img_buffer, format="PNG")
                zip_file.writestr(f"{image_name}_{weight}.png", img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# 汇总所有结果为一个总表格
def aggregate_results(images_to_process, selected_weights):
    all_features = []
    for uploaded_file in images_to_process:
        image = np.array(Image.open(uploaded_file))
        image_name = uploaded_file.name if hasattr(uploaded_file, "name") else "demo_image"
        for weight in selected_weights:
            model = load_model(weight)
            _, df_features = segment_image(model, image, image_name)
            # Add weight and image name to the DataFrame for clarity
            df_features['Model Weight'] = weight
            df_features['Image Name'] = image_name
            all_features.append(df_features)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame(columns=['Sample name', 'Loop number', 'Region area', 'Vessel density',
                                'Vessel area', 'Leading bud number', 'Perimeter', 'Aspect ratio',
                                'Solidity', 'Rectangularity', 'Compactness', 'Diameter',
                                'Irregularity', 'Vessel_length', 'Total_interval_length', 'Model Weight',
                                'Image Name'])

# 侧边栏收缩按钮
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = True  # 默认展开
if 'page' not in st.session_state:
    st.session_state.page = "Introduction"  # 默认页面

# 导航栏
with st.sidebar:
    st.title("导航")
    if st.button("介绍"):
        st.session_state.page = "Introduction"
        st.experimental_rerun()
    if st.button("分析"):
        st.session_state.page = "Model Segmentation"
        st.experimental_rerun()

# Home 按钮（右上角）
col1, col2 = st.columns([9, 1])  # 使用列布局将 Home 按钮推到右侧
with col1:
    st.write("")  # 占位符
with col2:
    if st.button("Home", key="home_btn"):
        st.session_state.page = "Introduction"
        st.experimental_rerun()

# 根据状态显示页面
if st.session_state.page == "Introduction":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>欢迎体验 Zebrafish AI</h1>
            <p style="font-size: 25px;">
                这是一个基于深度学习的交互式数据分析平台，专注于斑马鱼血管分割与分析<br>
                用户可以上传图片，选择不同的模型权重进行分割和分析<br>
                代码开源: <a href="https://github.com/chenjunzhou/Zebrafish-AI" target="_blank">https://github.com/chenjunzhou/Zebrafish-AI</a><br>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("开始分析", key="start_btn", help="点击进入模型分割页面"):
        st.session_state.page = "Model Segmentation"
        st.experimental_rerun()
    st.markdown(
        """
        <style>
        div.stButton > button:hover {
            background-color: white;
            border: none;
        }
        /* 增大“开始分析”按钮字体并去除边框 */
        #start_btn > button {
            font-size: 24px !important; /* 增大字体 */
            border: none !important; /* 去除边框 */
            padding: 15px 30px; /* 增加内边距以适配大字体 */
        }
        /* 自定义按钮样式 */
        div.stButton > button {
            background-color: white; /*##87CEEB*/
            color: black;
            padding: 15px 30px; /* 增大按钮内边距 */
            font-size: 24px; /* 增大按钮字体大小 */
            border: none !important; /* 强制移除边框 */
            border-radius: 5px;
            cursor: pointer;
            text-align: center; /* 确保按钮文字居中 */
        }
        div.stButton > button:hover {
            background-color: white;
        }
        /* 确保内容居中 */
        .centered-content {
            text-align: center;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 底部布局：左下角校徽，右下角联系信息
    st.markdown("<hr>", unsafe_allow_html=True)

    # 添加居中图片
    image_path = r"themes/4.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
                    <div class="centered-content">
                        <img src="data:image/jpeg;base64,{encoded_string}" width="800" style="display: block; margin: 0 auto;">
                        <p style="font-size: 22px; margin-top: 10px; text-align: center;">
                            <strong>Development of an Automated Morphometric Approach to Assess Vascular Outcomes</strong><br>
                            <strong>following Exposure to Environmental Chemicals in Zebrafish</strong><br>
                            <a href="https://doi.org/10.1289/EHP13214" target="_blank" style="font-size: 16px; text-decoration: underline; text-align: center; display: block;">https://doi.org/10.1289/EHP13214</a>
                        </p>
                    </div>
                    """,
            unsafe_allow_html=True
        )
    else:
        st.write("图片未找到，请检查 'themes/4.jpg' 是否存在")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        logo_path = r"themes/school.png"
        if os.path.exists(logo_path):
            image = Image.open(logo_path)
            st.image(image, width=100, caption="")
        else:
            st.write("校徽图片未找到'")

    with col_right:
        st.markdown(
            """
            <div style="text-align: right; font-size: 14px;">
                <p><strong>联系我们</strong><br>
                地址: 广州市中山二路74号<br>
                中山大学公共卫生学院</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif st.session_state.page == "Model Segmentation":
    st.write("# Model Segmentation and Analysis")

    # 选择模型权重
    selected_weights = st.multiselect("Select Model Weights", WEIGHTS_LIST, default=["CCV"])

    # 选择示例图像
    demo_images = glob.glob("images/*.bmp")
    demo_image_options = ["None"] + [os.path.basename(img) for img in demo_images]
    selected_demo_image = st.selectbox("Select Example Image", demo_image_options, index=0)

    # 上传图片
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "bmp"], accept_multiple_files=True)

    # 删除结果按钮
    if st.button("删除结果"):
        st.session_state.clear()
        st.session_state.page = "Model Segmentation"
        st.experimental_rerun()

    # 处理图片输入
    images_to_process = []
    if uploaded_files:
        images_to_process = [(file, file.name) for file in uploaded_files]  # 处理所有上传的图片
    elif selected_demo_image != "None":
        demo_image_path = os.path.join("images", selected_demo_image)
        file_object = open(demo_image_path, "rb")
        images_to_process = [(file_object, selected_demo_image)]

    if images_to_process and selected_weights:
        # 初始化存储结构
        all_results = {}  # 存储分割图像
        all_features = []  # 存储所有特征数据

        # 处理所有照片和模型权重
        for weight in selected_weights:
            model = load_model(weight)
            all_results[weight] = []
            for file_obj, image_name in images_to_process:
                image = np.array(Image.open(file_obj))
                segmented_image, df_features = segment_image(model, image, image_name)
                # 存储分割结果
                all_results[weight].append((image_name, segmented_image))
                # 添加模型权重和图片名称到特征数据
                df_features['Model Weight'] = weight
                df_features['Image Name'] = image_name
                all_features.append(df_features)

        # 展示逻辑：仅显示第一张照片的分割结果
        first_image_name = images_to_process[0][1]  # 第一张图片的名称
        st.write("### 分割预览")
        for i, weight in enumerate(selected_weights):
            if i < 4:  # 第一行：前四个权重
                if i == 0:
                    cols_row1 = st.columns(4)
                with cols_row1[i]:
                    st.write(f"**{weight}**")
                    for img_name, segmented_image in all_results[weight]:
                        if img_name == first_image_name:
                            st.image(segmented_image, caption=f"{img_name}", use_column_width=True)
                            break
            else:  # 第二行：后三个权重
                if i == 4:
                    st.write("#### ")
                    cols_row2 = st.columns(3)
                with cols_row2[i - 4]:
                    st.write(f"**{weight}**")
                    for img_name, segmented_image in all_results[weight]:
                        if img_name == first_image_name:
                            st.image(segmented_image, caption=f"{img_name}", use_column_width=True)
                            break

        # 汇总逻辑：生成包含所有照片的表格
        if all_features:
            total_df = pd.concat(all_features, ignore_index=True)
            st.write("### 分析结果汇总")
            st.dataframe(total_df)

        # 下载所有分割图片
        if st.button("下载分割图片"):
            zip_buffer = create_zip_of_results([file_obj for file_obj, _ in images_to_process], selected_weights)
            st.download_button(
                label="Download ZIP File",
                data=zip_buffer,
                file_name="segmented_results.zip",
                mime="application/zip"
            )

# 控制侧边栏显示状态
if not st.session_state.sidebar_state:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    st.write("")
