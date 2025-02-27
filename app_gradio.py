import argparse
import sys
from dataset.datasets import *
import segmentation_models_pytorch as smp
import torch
from tools import *
import errno
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

# DEVICE = 'cuda:0'
DEVICE = 'cpu'
dest_dir = './'
encoder='se_resnext50_32x4d'
encoder_weights='imagenet'
size=[416,1024]

REPO_ID = "mujita/eca_models"

brain_area_model=torch.load(
    hf_hub_download(REPO_ID, "brain_area_best_model_1024_416_cpu.pth")
)#/weights/brain_area/
# 把模型放到CPU上
brain_area_model.to('cpu')
brain_area_model.eval()

ccv_model=torch.load(
    hf_hub_download(REPO_ID, "CCV_best_model_1024_416_cpu.pth")
)#/weights/CCV/
ccv_model.to('cpu')
ccv_model.eval()

cv_model=torch.load(
    hf_hub_download(REPO_ID, "CV_best_model_1024_416_cpu.pth")
)#/weights/CV/
cv_model.to('cpu')
cv_model.eval()

cv2_model=torch.load(
    hf_hub_download(REPO_ID, "CV2_best_model_1024_416_cpu.pth")
)#/weights/CV2
cv2_model.to('cpu')
cv2_model.eval()

da_model=torch.load(
    hf_hub_download(REPO_ID, "DA_best_model_1024_416_cpu.pth")
)#/weights/DA
da_model.to('cpu')
da_model.eval()

isv_model=torch.load(
    hf_hub_download(REPO_ID, "ISV_best_model_1024_416_cpu.pth")
)#/weights/ISV
isv_model.to('cpu')
isv_model.eval()

macrovasculature_model=torch.load(
    hf_hub_download(REPO_ID, "macrovasculature_best_model_1024_416_cpu.pth")
)#/weights/macrovasculature
macrovasculature_model.to('cpu')
macrovasculature_model.eval()

pcv_model=torch.load(
    hf_hub_download(REPO_ID, "PCV_best_model_1024_416_cpu.pth")
)#/weights/PCV
pcv_model.to('cpu')
pcv_model.eval()

siv_model=torch.load(
    hf_hub_download(REPO_ID, "SIV_best_model_1024_416_cpu.pth")
)#/weights/SIV
siv_model.to('cpu')
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
    if weight in weights_dict.keys():
        model = weights_dict[weight]
        return_images = []
        if not images:
            return None
        for image, _ in images:
            return_image, return_name = predict_area(image, model)
            return_images.append((return_image, return_name))
        return return_images
    else:
        return None

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## Zebrafish Blood Vessel Segmentation")
    gr.Interface(images_process, [gr.Gallery(), gr.Dropdown(weights_list, value="CCV")],
                 gr.Gallery(), submit_btn="Segment", clear_btn="clear outs", allow_flagging=False, examples=[[[r'./images/0.bmp/', r'./images/1.bmp/', r'./images/2.bmp/', r'./images/3.bmp/']]])
    #gr.Interface(images_process, [gr.Gallery(), gr.Dropdown(weights_list, value="CCV")], gr.Gallery(), submit_btn="Segment", clear_btn="clear outs", allow_flagging=False)

#demo.launch(server_name="0.0.0.0", server_port=1234, share=False)
if __name__=="__main__":
    demo.launch()
