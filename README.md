# MCAM
This code is for our paper named MultiModal causal anaysis model for Driving Video Understanding.
And the paper has been accept in ICCV 2025.

This Work is based on the swin-video-transformer and ADAPT. And the CAM is inspired from LLCP, Thanks for them superior work, the cite is as following.

@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}
_____________________________________
@article{jin2023adapt,
  title={ADAPT: Action-aware Driving Caption Transformer},
  author={Jin, Bu and Liu, Xinyu and Zheng, Yupeng and Li, Pengfei and Zhao, Hao and Zhang, Tong and Zheng, Yuhang and Zhou, Guyue and Liu, Jingjing},
  journal={arXiv preprint arXiv:2302.00673},
  year={2023}
}
________________________________________
@inproceedings{chen2024llcp,
  title={LLCP: Learning Latent Causal Processes for Reasoning-based Video Question Answer},
  author={Chen, Guangyi and Li, Yuke and Liu, Xiao and Li, Zijian and Al Suradi, Eman and Wei, Donglai and Zhang, Kun},
  booktitle={ICLR},
  year={2024}
}
________________________________________


our environment setting is likely, as following.

First, we need install the anoconda and pytorch.
``` python
conda create --name MCAM python=3.8
```

``` python
conda activate MCAM
```
Install Pytorch torch版本可以按照自己的设备进行调整，只要符合torch本身的架构要求就好
``` python
pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```
Install apex 可以选择手动下载 apex的zip包，然后解压到指定文件夹下
``` python
#git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```
Install mpi4py 安装必要的包
``` python
conda install -c conda-forge mpi4py openmpi
```
安装其他的依赖项，依赖项中有不符合的，在运行过程中，缺少的部分 使用 pip install --对应名字 即可 
``` python
pip install -r requirements.txt
```
这里给出文件夹的分布情况

${REPO_DIR}
|-- checkpoints
|-- datasets  
|   |-- BDDX
|   |   |-- frame_tsv
|   |   |-- captions_BDDX.json
|   |   |-- training_32frames_caption_coco_format.json
|   |   |-- training_32frames.yaml
|   |   |-- training.caption.lineidx
|   |   |-- training.caption.lineidx.8b
|   |   |-- training.caption.linelist.tsv
|   |   |-- training.caption.tsv
|   |   |-- training.img.lineidx
|   |   |-- training.img.lineidx.8b
|   |   |-- training.img.tsv
|   |   |-- training.label.lineidx
|   |   |-- training.label.lineidx.8b
|   |   |-- training.label.tsv
|   |   |-- training.linelist.lineidx
|   |   |-- training.linelist.lineidx.8b
|   |   |-- training.linelist.tsv
|   |   |-- validation...
|   |   |-- ...
|   |   |-- validation...
|   |   |-- testing...
|   |   |-- ...
|   |   |-- testing...
|-- datasets_part
|-- docs
|-- models
|   |-- basemodel
|   |-- captioning
|   |-- video_swin_transformer
|-- scripts 
|-- src
|-- README.md 
|-- ... 
|-- ... 



We will upload our code one by one, due to the file which could not be move and adjust online.
