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
DEVICE = 'cpu'


encoder='se_resnext50_32x4d'
encoder_weights='imagenet'
size=[416,1024]
# model=Unet()
model=(torch.load(r'C:\Users\Mu\datascience\gradio\weights\CCV\CCV_best_model_1024_416.pth',
                 map_location='cuda:0'))
image=(r'D:\anaconda3\ECA-ResXUnet\images\image_0.png')
model.eval()
print(type(model))
# if isinstance(model, torch.nn.DataParallel):
#     model = model.module
# torch.device('cpu'))
def cv_imread(file_path):
    with open(file_path, 'rb') as f:
        img_bytes = bytearray(f.read())
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, -1)
def predict_CCV(img, model):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    img_org = cv_imread(img)

    x_tensor = torch.from_numpy(img_org).to(DEVICE).unsqueeze(0)
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
    os.makedirs(os.path.join(dest_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'img_region'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'mask_region'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'img_seg'), exist_ok=True)
    # save image region
    cv2.imencode(name[-4:], img_masked)[1].tofile()



# iface = gr.Interface(
#     fn=predict_CCV(image,model),
#     inputs=[
#         gr.File(label="Upload Image"),                                    # 上传图片输入
#         gr.CheckboxGroup(['CCV'], label="Region List"),                   # 区域列表选项
#     ],
#     outputs=gr.File(label="Download Results"),                    # 设置输出结果为可下载的文件
#     title="Segmentation Model Prediction",
#     description="Select regions and upload an image to segment. Download results as a zip file."
# )

if __name__ == "__main__":

    # iface.launch(share=True)

