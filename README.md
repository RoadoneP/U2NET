# Clothes Segmentation using U2NET #

This repo contains training code, inference code and pre-trained model for Cloths Parsing from human portrait.</br>
Here clothes are parsed into 3 category: Upper body(red), Lower body(green) and Full body(yellow)



This model works well with any background and almost all poses. For more samples visit [samples.md](samples.md)

## Result

### success
<img src="images\1.png" width = 300> <img src="images\1-1.png" width = 300>
<img src="images\2.png" width = 300> <img src="images\2-1.png" width = 300>
<img src="images\3.png" width = 300> <img src="images\3-1.png" width = 300>
<img src="images\5.png" width = 300> <img src="images\5-1.png" width = 300>

### false
<img src="images\4.png" width = 300> <img src="images\4-1.png" width = 300>



# Techinal details

* **U2NET** : This project uses an amazing [U2NET](https://arxiv.org/abs/2005.09007) as a deep learning model. Instead of having 1 channel output from u2net for typical salient object detection task it outputs 4 channels each respresting upper body cloth, lower body cloth, fully body cloth and background. Only categorical cross-entropy loss is used for a given version of the checkpoint.

* **Dataset** : U2net is trained on 45k images [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) dataset. To reduce complexity, I have clubbed the original 42 categories from dataset labels into 3 categories (upper body, lower body and full body). All images are resized into square `¯\_(ツ)_/¯` 768 x 768 px for training. (This experiment was conducted with 768 px but around 384 px will work fine too if one is retraining on another dataset).

# Training 

- For training this project requires,
<ul>
    <ul>
    <li>&nbsp; PyTorch > 1.3.0</li>
    <li>&nbsp; tensorboardX</li>
    <li>&nbsp; gdown</li>
    </ul>
</ul>

- Download dataset from this [link](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data), extract all items.
- Set path of `train` folder which contains training images and `train.csv` which is label csv file in `options/base_options.py`
- To port original u2net of all layer except last layer please run `python setup_model_weights.py` and it will generate weights after model surgey in `prev_checkpoints` folder.
- You can explore various options in `options/base_options.py` like checkpoint saving folder, logs folder etc.
- For single gpu set `distributed = False` in `options/base_options.py`, for multi gpu set it to `True`.
- For single gpu run `python train.py`
- For multi gpu run <br>
&nbsp;`python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --use_env train.py` <br>
Here command is for single node, 4 gpu. Tested only for single node.
- You can watch loss graphs and samples in tensorboard by running tensorboard command in log folder.


# Testing/Inference
- Download pretrained model from this [link](https://drive.google.com/file/d/1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ/view?usp=sharing)(165 MB) in `trained_checkpoint` folder.
- Put input images in `input_images` folder
- Run `python infer.py` for inference.
- Output will be saved in `output_images`


# Acknowledgements
- U2net model is from original [u2net repo](https://github.com/xuebinqin/U-2-Net). Thanks to Xuebin Qin for amazing repo.
- Complete repo follows structure of [Pix2pixHD repo](https://github.com/NVIDIA/pix2pixHD)

