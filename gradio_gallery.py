import argparse
import sys
from datasets.datasets import *
import segmentation_models_pytorch as smp
import torch
from tools import *
import errno
import gradio as gr
import numpy as np

#DEVICE = 'cuda:0'
DEVICE = "cpu"
dest_dir = './'
encoder='se_resnext50_32x4d'
encoder_weights='imagenet'
size=[416,1024]

brain_area_model=torch.load(r'./weights/brain_area/brain_area_best_model_1024_416.pth',map_location=DEVICE)
brain_area_model.eval()

ccv_model=torch.load(r'./weights/CCV/CCV_best_model_1024_416.pth',map_location=DEVICE)
ccv_model.eval()


cv_model=torch.load(r'./weights/CV/CV_best_model_1024_416.pth',map_location=DEVICE)
cv_model.eval()

cv2_model=torch.load(r'./weights/CV2/CV2_best_model_1024_416.pth',map_location=DEVICE)
cv2_model.eval()

da_model=torch.load(r'./weights/DA/DA_best_model_1024_416.pth',map_location=DEVICE)
da_model.eval()

isv_model=torch.load(r'./weights/ISV/ISV_best_model_1024_416.pth',map_location=DEVICE)
isv_model.eval()

macrovasculature_model=torch.load(r'./weights/macrovasculature/macrovasculature_best_model_1024_416.pth',map_location=DEVICE)
macrovasculature_model.eval()

pcv_model=torch.load(r'./weights/PCV/PCV_best_model_1024_416.pth',map_location=DEVICE)
pcv_model.eval()

siv_model=torch.load(r'./weights/SIV/SIV_best_model_1024_416.pth',map_location=DEVICE)
siv_model.eval()

os.makedirs(os.path.join(dest_dir, 'mask'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'img_region'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'mask_region'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'img_seg'), exist_ok=True)
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
augmentation=get_validation_augmentation(size)
preprocessing=get_preprocessing(preprocessing_fn)

weights_list = ["brain_area", "CCV", "CV", "CV2", "DA", "ISV", "macrovasculature", "PCV", "SIV"]

weights_dict = {"brain_area":brain_area_model, "CCV":ccv_model, 
                "CV":cv_model, "CV2":cv2_model, "DA":da_model, 
                "ISV":isv_model, "macrovasculature":macrovasculature_model, 
                "PCV":pcv_model, "SIV":siv_model}


def preprocessing_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample = augmentation(image=image)
    image = sample['image']

    sample = preprocessing(image=image)
    image = sample['image']

    return image

def cv_imread(file_path):
    with open(file_path, 'rb') as f:
        img_bytes = bytearray(f.read())
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, -1)

def predict_area(img, model):
    with torch.no_grad():
        name = os.path.basename(img)
        img_path = img
        img_org = cv_imread(img)
        x_tensor = torch.from_numpy(preprocessing_img(img_org)).to(DEVICE).unsqueeze(0)
        pr_mask = model(x_tensor)
        # pr_mask =np.argmax(model.predict(x_tensor), axis=1)
        pr_mask = (pr_mask.squeeze().cpu().numpy())
        pr_mask_org_size = cv2.resize(pr_mask, (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_CUBIC)
        pr_mask_org_size_threshold = pr_mask_org_size.round().astype(np.uint8) * 255
            # draw bbox
        contours, hierarchy = cv2.findContours(pr_mask_org_size_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # apply mask on the original image and extract out rect of detected region
        cnt = get_largest_contour(contours)
        mask = make_mask_by_contours(img_org.shape, img_org.dtype, [cnt])
        img_masked = np.where(mask, img_org, 0)
        mask_rect = extract_contour_rect(pr_mask_org_size_threshold, cnt)
        # output the img_rect and mask_rect
        img_rect = extract_contour_rect(img_masked, cnt)

        # save mask
        cv2.imencode(name[-4:], pr_mask_org_size_threshold)[1].tofile(os.path.join(dest_dir, 'mask', os.path.basename(img_path)))
        cv2.imencode(name[-4:], mask_rect)[1].tofile(os.path.join(dest_dir, 'mask_region', os.path.basename(img_path)))

        # save image region
        cv2.imencode(name[-4:], img_rect)[1].tofile(
            os.path.join(dest_dir,'img_region', os.path.basename(img_path)))
        cv2.imencode(name[-4:], img_masked)[1].tofile(
            os.path.join(dest_dir, 'img_seg', os.path.basename(img_path)))
        
        return os.path.join(dest_dir,'img_seg', os.path.basename(img_path)), name

def images_process(images, weight):
    model = weights_dict[weight]
    return_images = []
    if not images:
        return None
    for image, _ in images:
        return_image, return_name = predict_area(image, model)
        return_images.append((return_image, return_name))
    return return_images



with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## 斑马鱼组织分割系统")
    gr.Interface(images_process, [gr.Gallery(), gr.Dropdown(weights_list, value="CCV")], gr.Gallery(), 
                 submit_btn="分割", clear_btn="清除结果", allow_flagging=False, 
                 examples=[[["./img_region/0.bmp"]]])

demo.launch(server_name="0.0.0.0", server_port=8061)