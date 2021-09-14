import os

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks.u2net import U2NET, U2NETP 

device = "cpu"

image_dir = "input_images"
result_dir = "output_images"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
do_palette = False


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

images_list = sorted(os.listdir(image_dir))
pbar = tqdm(total=len(images_list))
for image_name in images_list:
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    # ([1, 4, size, size]) 1개의 배치에 4개 class에 대한 확률값 
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    # ([1, 4, size, size]) 1개의 배치에 4개 class에 대한 확률값 중 최대 고르기
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    output_arr =np.repeat(output_arr[:, :, np.newaxis], 3, axis=2)
    output_arr[np.where((output_arr==[1,1,1]).all(axis=2))] = [0, 255, 0]
    output_arr[np.where((output_arr==[2,2,2]).all(axis=2))] = [255, 0, 0]
    output_arr[np.where((output_arr==[3,3,3]).all(axis=2))] = [0, 0, 255]
    output_img = Image.fromarray(output_arr.astype("uint8"), mode="RGB")

    blended = Image.blend(output_img, img, alpha=0.5)    
    blended.save(os.path.join(result_dir, image_name[:-3] + "png"))
    output_img.save(os.path.join(result_dir, "1" + image_name[:-3] + "png"))
    pbar.update(1)

pbar.close()