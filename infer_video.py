import os

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = "cuda"

checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")

image_dir = "input_images"
result_dir = "output_images"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
do_palette = True
color_dir = 'color_images/purple.jpg'

def extract_mask(output_arr, img):
  original_img = np.array(img)
  background_mask = np.where(output_arr==0, 1, 0)
  class1_mask = np.where(output_arr==1, 1, 0) 
  class2_mask = np.where(output_arr==2, 1, 0)
  class3_mask = np.where(output_arr==3, 1, 0)
  background_mask =np.repeat(background_mask[:, :, np.newaxis], 3, axis=2) * original_img
  class1_mask =np.repeat(class1_mask[:, :, np.newaxis], 3, axis=2) * original_img
  class2_mask =np.repeat(class2_mask[:, :, np.newaxis], 3, axis=2) * original_img
  class3_mask =np.repeat(class3_mask[:, :, np.newaxis], 3, axis=2) * original_img
  
  masks = [background_mask, class1_mask, class2_mask, class3_mask]
  return masks
  
def color_change(background_seg_imgs, mask, color_dir):
  one_color = cv2.imread(color_dir) # 이미지 파일을 컬러로 불러옴
  height, width = one_color.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

  one_hsv = cv2.cvtColor(one_color, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환
  center_h = (height // 2)
  center_w = (width // 2)

  one_h = one_hsv[center_h][center_w][0]
  one_s = one_hsv[center_h][center_w][1]
  one_v = one_hsv[center_h][center_w][2]

  test = mask
  hsv = cv2.cvtColor(test.astype("uint8"), cv2.COLOR_RGB2HSV)

  (h, s, v) = cv2.split(hsv)

  eval_s = s.sum() // np.count_nonzero(s)
  eval_v = v.sum() // np.count_nonzero(v)


  h[:, :] = one_h
  s[:,:] = one_s
 # s[:, :] = np.where(s + (one_s - eval_s) > 255, 255, s + (one_s - eval_s))
  #v[:, :] = np.where(v + (one_v - eval_v) > 255, 255, v + (one_v - eval_v))


  hsv = cv2.merge((h, s, v))

  rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  
  test = rgb * np.where(mask!=0, 1, 0) 
  test = test + background_seg_imgs
  return test
  


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()



VideoSignal = cv2.VideoCapture(0)

while True:
    ret, frame = VideoSignal.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    background_mask, class1_mask, class2_mask, class3_mask = extract_mask(output_arr, img)
    output_img = color_change(background_mask + class2_mask, class1_mask, color_dir)
    output_img = np.array(output_img, dtype=np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
VideoSignal.release()

cv2.destroyAllWindows()