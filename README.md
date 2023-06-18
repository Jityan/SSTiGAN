# Enhanced Text-to-Image Synthesis With Self-Supervision

This repository provides the **pytorch** code for the paper "[Enhanced Text-to-Image Synthesis With Self-Supervision](https://doi.org/10.1109/ACCESS.2023.3268869)" by Yong Xuan Tan, Chin Poo Lee, Mai Neo, Kian Ming Lim, Jit Yan Lim.

<img src="figures/framework.jpg" width="850px" />

## Environment
The code is tested on Windows 10 with Anaconda3 and following packages:
- python 3.7.11
- pytorch 1.9.0
- torchvision 0.10.0

## Dataset
We follow the same procedure and structure as [SSTIS](https://github.com/Jityan/SSTIS).

Download the preprocessed char-CNN-RNN text embeddings for [flowers](https://www.dropbox.com/sh/g8rmz41xblaszb1/AABPNtIcLu1fKNoBsJTHJTIKa?dl=0) and [birds](https://www.dropbox.com/sh/v0vcgwue2nkwgrf/AACxoRYTAAacmPVfEvY-eDzia?dl=0) and the images for [flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and [birds](http://www.vision.caltech.edu/datasets/cub_200_2011/). Put them into `./data/oxford` and `./data/cub` folder.

## Experiments
To train on Oxford:<br/>
```
python main.py --dataset flowers --exp_num oxford_exp
```
To evaluate on Oxford:<br/>
```
python main.py --dataset flowers --exp_num oxford_exp --is_test true
```

## Pre-trained Models
Download the [pretrained models](https://drive.google.com/file/d/1ovCIhapyRazGIda0dw6okLeqtU48MiW2/view?usp=drive_link). Extract it to the `saved_model` folder.

Examples generated by SS-TiGAN:
<img src="figures/examples.jpg" width="850px" />

## Citation
If you find this repo useful for your research, please consider citing the paper:
```
@ARTICLE{10105864,
  author={Tan, Yong Xuan and Lee, Chin Poo and Neo, Mai and Lim, Kian Ming and Lim, Jit Yan},
  journal={IEEE Access}, 
  title={Enhanced Text-to-Image Synthesis With Self-Supervision}, 
  year={2023},
  volume={11},
  number={},
  pages={39508-39519},
  doi={10.1109/ACCESS.2023.3268869}
}
```

## Contacts
For any questions, please contact: <br/>

Yong Xuan Tan (1141124379@student.mmu.edu.my) <br/>
Jit Yan Lim (lim.jityan@mmu.edu.my)

## Acknowlegements
- [Text-to-Image Synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis)
- [StackGAN](https://github.com/hanzhanggit/StackGAN)
- [StackGAN++](https://github.com/hanzhanggit/StackGAN-v2)
- [HDGAN](https://github.com/ypxie/HDGan)
- [SS-GAN](https://github.com/vandit15/Self-Supervised-Gans-Pytorch)

## License
This code is released under the MIT License (refer to the LICENSE file for details).