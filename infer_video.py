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
do_palette = True


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

palette = get_palette(4)

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
    output_arr = np.array(output_arr, dtype=np.uint8)
    output_arr =np.repeat(output_arr[:, :, np.newaxis], 3, axis=2)
    output_arr[np.where((output_arr==[1,1,1]).all(axis=2))] = [0, 255, 0]
    output_arr[np.where((output_arr==[2,2,2]).all(axis=2))] = [255, 0, 0]
    output_arr[np.where((output_arr==[3,3,3]).all(axis=2))] = [0, 0, 255]
    output_img = Image.fromarray(output_arr.astype("uint8"), mode="RGB")
    frame = Image.fromarray(frame, mode="RGB")

    blended = Image.blend(output_img, frame, alpha=0.5)
    result = np.array(blended)
    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
VideoSignal.release()

cv2.destroyAllWindows()