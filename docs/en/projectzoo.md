# Welcome to Projects of MMAction2

In this folder, we welcome all contributions of deep-learning video understanding models from the community.

Here, these requirements, e.g., code standards, are not as strict as in the core package. Thus, developers from the community can implement their algorithms much more easily and efficiently in MMAction2. We appreciate all contributions from the community to make MMAction2 greater.

Here is an [example project](./example_project) about how to add your algorithms easily.

We also provide some documentation listed below:

- [Contribution Guide](https://mmaction2.readthedocs.io/en/latest/get_started/contribution_guide.html)

  The guides for new contributors about how to add your projects to MMAction2.

- [Discussions](https://github.com/open-mmlab/mmaction2/discussions)

  Welcome to start a discussion!
# Example Project

This is an example README for community `projects/`. You can write your README in your own project. Here are
some recommended parts of a README for others to understand and use your project, you can copy or modify them
according to your project.

## Usage

### Setup Environment

Please refer to [Get Started](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the Kinetics400 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md).

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc |  testing protocol  |                    config                     |                                   ckpt |                            log |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :-------------------------------------------: | -------------------------------------: | -----------------------------: |
|          1x1x3          |  224x224   |  8   | ResNet50 | ImageNet |  72.83   |  90.65   | 25 clips x 10 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/example_project/configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://example/checkpoint/url) | [log](https://example/log/url) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@misc{2020mmaction2,
  title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
  author={MMAction2 Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
  year={2020}
}
```

## Checklist

Here is a checklist of this project's progress, and you can ignore this part if you don't plan to contribute to MMAction2 projects.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmaction.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. Usually, ±0.1% is acceptable for the action recognition task on Kinetics400. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmaction2/blob/main/tests/models/backbones/test_resnet.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for MMAction2 to acquire your models. [Example](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/README.md) -->
# ActionCLIP Project

[ActionCLIP: A New Paradigm for Video Action Recognition](https://arxiv.org/abs/2109.08472)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The canonical approach to video action recognition dictates a neural model to do a classic and standard 1-of-N majority vote task. They are trained to predict a fixed set of predefined categories, limiting their transferable ability on new datasets with unseen concepts. In this paper, we provide a new perspective on action recognition by attaching importance to the semantic information of label texts rather than simply mapping them into numbers. Specifically, we model this task as a video-text matching problem within a multimodal learning framework, which strengthens the video representation with more semantic language supervision and enables our model to do zero-shot action recognition without any further labeled data or parameters requirements. Moreover, to handle the deficiency of label texts and make use of tremendous web data, we propose a new paradigm based on this multimodal learning framework for action recognition, which we dub "pre-train, prompt and fine-tune". This paradigm first learns powerful representations from pre-training on a large amount of web image-text or video-text data. Then it makes the action recognition task to act more like pre-training problems via prompt engineering. Finally, it end-to-end fine-tunes on target datasets to obtain strong performance. We give an instantiation of the new paradigm, ActionCLIP, which not only has superior and flexible zero-shot/few-shot transfer ability but also reaches a top performance on general action recognition task, achieving 83.8% top-1 accuracy on Kinetics-400 with a ViT-B/16 as the backbone.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/58767402/237413093-75d76018-0521-4642-af68-32141fb4fed1.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2. Run the following command to install `clip`.

```shell
pip install git+https://github.com/openai/CLIP.git
```

Assume that you are located at `$MMACTION2/projects/actionclip`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the Kinetics400 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### Kinetics400

| frame sampling strategy | backbone | top1 acc | top5 acc |  testing protocol  |                                config                                |                                ckpt                                 |
| :---------------------: | :------: | :------: | :------: | :----------------: | :------------------------------------------------------------------: | :-----------------------------------------------------------------: |
|          1x1x8          | ViT-B/32 |   77.6   |   93.8   | 8 clips  x 1 crop  | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/configs/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/vit-b-32-8f.pth)\[1\] |
|          1x1x8          | ViT-B/16 |   80.3   |   95.2   | 8 clips  x 1 crop  | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/configs/actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb/vit-b-16-8f.pth)\[1\] |
|         1x1x16          | ViT-B/16 |   81.1   |   95.6   | 16 clips  x 1 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/configs/actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb/vit-b-16-16f.pth)\[1\] |
|         1x1x32          | ViT-B/16 |   81.3   |   95.8   | 32 clips  x 1 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/configs/actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb/vit-b-16-32f.pth)\[1\] |

\[1\] The models are ported from the repo [ActionCLIP](https://github.com/sallymmx/ActionCLIP) and tested on our data. Currently, we only support the testing of ActionCLIP models. Due to the variation in testing data, our reported test accuracy differs from that of the original repository (on average, it is lower by one point). Please refer to this [issue](https://github.com/sallymmx/ActionCLIP/issues/14) for more details.

### Kinetics400 (Trained on Our K400 dataset)

| frame sampling strategy | gpus | backbone | top1 acc | top5 acc | testing protocol  |                    config                     |                     ckpt                     |                     log                     |
| :---------------------: | :--: | :------: | :------: | :------: | :---------------: | :-------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|          1x1x8          |  8   | ViT-B/32 |   77.5   |   93.2   | 8 clips  x 1 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb_20230801-8535b794.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.log) |
|          1x1x8          |  8   | ViT-B/16 |   81.3   |   95.2   | 8 clips  x 1 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/configs/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb_20230801-b307a0cd.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb.log) |

## Zero-Shot Prediction

We offer two methods for zero-shot prediction as follows. The `test.mp4` can be downloaded from [here](https://github-production-user-asset-6210df.s3.amazonaws.com/58767402/237333525-89ebee9a-573e-4e27-9047-0ad6422fa82f.mp4).

### Using Naive Pytorch

```python
import torch
import clip
from models.load import init_actionclip
from mmaction.utils import register_all_modules

register_all_modules(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = init_actionclip('ViT-B/32-8', device=device)

video_anno = dict(filename='test.mp4', start_index=0)
video = preprocess(video_anno).unsqueeze(0).to(device)

template = 'The woman is {}'
labels = ['singing', 'dancing', 'performing']
text = clip.tokenize([template.format(label) for label in labels]).to(device)

with torch.no_grad():
    video_features = model.encode_video(video)
    text_features = model.encode_text(text)

video_features /= video_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100 * video_features @ text_features.T).softmax(dim=-1)
probs = similarity.cpu().numpy()

print("Label probs:", probs)  # [[9.995e-01 5.364e-07 6.666e-04]]
```

### Using MMAction2 APIs

```python
import mmengine
import torch
from mmaction.utils import register_all_modules
from mmaction.apis import inference_recognizer, init_recognizer

register_all_modules(True)

config_path = 'configs/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb.py'
checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/vit-b-32-8f.pth'
template = 'The woman is {}'
labels = ['singing', 'dancing', 'performing']

# Update the labels, the default is the label list of K400.
config = mmengine.Config.fromfile(config_path)
config.model.labels_or_label_file = labels
config.model.template = template

device = "cuda" if torch.cuda.is_available() else "cpu"
model = init_recognizer(config=config, checkpoint=checkpoint_path, device=device)

pred_result = inference_recognizer(model, 'test.mp4')
probs = pred_result.pred_score.cpu().numpy()
print("Label probs:", probs)  # [9.995e-01 5.364e-07 6.666e-04]
```

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@article{wang2021actionclip,
  title={Actionclip: A new paradigm for video action recognition},
  author={Wang, Mengmeng and Xing, Jiazheng and Liu, Yong},
  journal={arXiv preprint arXiv:2109.08472},
  year={2021}
}
```
# CTRGCN Project

[Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Graph convolutional networks (GCNs) have been widely used and achieved remarkable results in skeleton-based action recognition. In GCNs, graph topology dominates feature aggregation and therefore is the key to extracting representative features. In this work, we propose a novel Channel-wise Topology Refinement Graph Convolution (CTR-GC) to dynamically learn different topologies and effectively aggregate joint features in different channels for skeleton-based action recognition. The proposed CTR-GC models channel-wise topologies through learning a shared topology as a generic prior for all channels and refining it with channel-specific correlations for each channel. Our refinement method introduces few extra parameters and significantly reduces the difficulty of modeling channel-wise topologies. Furthermore, via reformulating graph convolutions into a unified form, we find that CTR-GC relaxes strict constraints of graph convolutions, leading to stronger representation capability. Combining CTR-GC with temporal modeling modules, we develop a powerful graph convolutional network named CTR-GCN which notably outperforms state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/58767402/223147561-9158fd51-8963-47c9-9338-de70470820cc.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

Assume that you are located at `$MMACTION2/projects/ctrgcn`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the NTU60 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### NTU60_XSub_2D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  CTRGCN  |   89.6   |     10 clips     | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/ctrgcn/configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230308-7aba454e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |

### NTU60_XSub_3D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  CTRGCN  |   89.0   |     10 clips     | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/ctrgcn/configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20230308-950dca0a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@inproceedings{chen2021channel,
  title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={CVPR},
  pages={13359--13368},
  year={2021}
}
```
# Gesture Recognition

<!-- [ALGORITHM] -->

## Introduction

<!-- [ABSTRACT] -->

In this project, we present a skeleton based pipeline for gesture recognition. The pipeline is three-stage. The first stage consists of a hand detection module that outputs bounding boxes of human hands from video frames. Afterwards, the second stage employs a pose estimation module to generate keypoints of the detected hands. Finally, the third stage utilizes a skeleton-based gesture recognition module to classify hand actions based on the provided hand skeleton. The three-stage pipeline is lightweight and can achieve real-time on CPU devices. In this README, we provide the models and the inference demo for the project. Training data preparation and training scripts are described in [TRAINING.md](https://github.com/open-mmlab/mmaction2/blob/main/projects/gesture_recognition/TRAINING.md).

## Hand detection stage

Hand detection results on OneHand10K validation dataset

| Config                                                  | Input Size | bbox mAP | bbox mAP 50 | bbox mAP 75 |                         ckpt                          |                         log                          |
| :------------------------------------------------------ | :--------: | :------: | :---------: | :---------: | :---------------------------------------------------: | :--------------------------------------------------: |
| [rtmdet_nano](https://github.com/open-mmlab/mmaction2/blob/main/projects/gesture_recognition/configs/rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320.py) |  320x320   |  0.8100  |   0.9870    |   0.9190    | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320_20230524-f6ffed6a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320.log) |

## Pose estimation stage

Pose estimation results on COCO-WholeBody-Hand validation set

| Config                                                                                                 | Input Size | PCK@0.2 |  AUC  | EPE  |                  ckpt                   |
| :----------------------------------------------------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :-------------------------------------: |
| [rtmpose_m](https://github.com/open-mmlab/mmaction2/blob/main/projects/gesture_recognition/configs/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.815  | 0.837 | 4.51 | [ckpt](https://download.openmmlab.com/) |

## Gesture recognition stage

Skeleton base gesture recognition results on Jester validation

| Config                                                  | Input Size | Top 1 accuracy | Top 5 accuracy |                          ckpt                          |                          log                          |
| :------------------------------------------------------ | :--------: | :------------: | :------------: | :----------------------------------------------------: | :---------------------------------------------------: |
| [STGCNPP](https://github.com/open-mmlab/mmaction2/blob/main/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.py) |  100x17x3  |     89.22      |     97.52      | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d_20230524-fffa7ff0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.log) |
# Knowledge Distillation Based on MMRazor

Knowledge Distillation is a classic model compression method. The core idea is to "imitate" a teacher model (or multi-model ensemble) with better performance and more complex structure by guiding a lightweight student model, improving the performance of the student model without changing its structure. [MMRazor](https://github.com/open-mmlab/mmrazor) is a model compression toolkit for model slimming and AutoML, which supports several KD algorithms. In this project, we take TSM-MobileNetV2 as an example to show how to use MMRazor to perform knowledge distillation on action recognition models. You could refer to more [MMRazor](https://github.com/open-mmlab/mmrazor) for more model compression algorithms.

## Description

This is an implementation of MMRazor Knowledge Distillation Application, we provide action recognition configs and models for MMRazor.

## Usage

### Prerequisites

- [MMRazor v1.0.0](https://github.com/open-mmlab/mmrazor/tree/v1.0.0) or higher

There are two install modes:

Option (a). Install as a Python package

```shell
mim install "mmrazor>=1.0.0"
```

Option (b). Install from source

```shell
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
pip install -v -e .
```

### Setup Environment

Please refer to [Get Started](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

### Data Preparation

Prepare the Kinetics400 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Training commands

**To train with single GPU:**

```bash
mim train mmrazor configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py
```

**To train with multiple GPUs:**

```bash
mim train mmrazor configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmrazor configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

Please convert the knowledge distillation checkpoint to student-only checkpoint with following commands, you will get a checkpoint with a '\_student.pth' suffix under the same directory as the original checkpoint. Then take the student-only checkpoint for testing.

```bash
mim run mmrazor convert_kd_ckpt_to_student $CHECKPOINT
```

**To test with single GPU:**

```bash
mim test mmaction tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results and models

| Location |   Dataset    |    Teacher     |      Student      |     Acc     | Acc(T) | Acc(S) |        Config         | Download                                                                      |
| :------: | :----------: | :------------: | :---------------: | :---------: | :----: | :----: | :-------------------: | :---------------------------------------------------------------------------- |
|  logits  | Kinetics-400 | [TSM-ResNet50] | [TSM-MobileNetV2] | 69.60(+0.9) | 73.22  | 68.71  | [config][distill_tsm] | [teacher][tsm_r50_pth] \| [model][distill_pth_tsm] \| [log][distill_log_tsm]  |
|  logits  | Kinetics-400 |   [TSN-Swin]   |  [TSN-ResNet50]   | 75.54(+1.4) | 79.22  | 74.12  | [config][distill_tsn] | [teacher][tsn_swin_pth] \| [model][distill_pth_tsn] \| [log][distill_log_tsn] |

## Citation

```latex
@article{huang2022knowledge,
  title={Knowledge Distillation from A Stronger Teacher},
  author={Huang, Tao and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal={arXiv preprint arXiv:2205.10536},
  year={2022}
}
```

[distill_log_tsm]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.log
[distill_log_tsn]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsn-swin_tsn-r50_1x1x8_k400/kd_logits_tsn-swin_tsn-r50_1x1x8_k400.log
[distill_pth_tsm]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400_20230517-c3e8aa0d.pth
[distill_pth_tsn]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsn-swin_tsn-r50_1x1x8_k400/kd_logits_tsn-swin_tsn-r50_1x1x8_k400_student_20230530-f938d404.pth
[distill_tsm]: https://github.com/open-mmlab/mmaction2/blob/main/projects/knowledge_distillation/configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py
[distill_tsn]: https://github.com/open-mmlab/mmaction2/blob/main/projects/knowledge_distillation/configs/kd_logits_tsn-swin_tsn-r50_8xb16_k400.py
[tsm-mobilenetv2]: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py
[tsm-resnet50]: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py
[tsm_r50_pth]: https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb_20220831-a6db1e5d.pth
[tsn-resnet50]: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py
[tsn-swin]: https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb.py
[tsn_swin_pth]: https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb_20230530-428f0064.pth
# MSG3D Project

[Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition](https://arxiv.org/abs/2003.14111)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Spatial-temporal graphs have been widely used by skeleton-based action recognition algorithms to model human action dynamics. To capture robust movement patterns from these graphs, long-range and multi-scale context aggregation and spatial-temporal dependency modeling are critical aspects of a powerful feature extractor. However, existing methods have limitations in achieving (1) unbiased long-range joint relationship modeling under multi-scale operators and (2) unobstructed cross-spacetime information flow for capturing complex spatial-temporal dependencies. In this work, we present (1) a simple method to disentangle multi-scale graph convolutions and (2) a unified spatial-temporal graph convolutional operator named G3D. The proposed multi-scale aggregation scheme disentangles the importance of nodes in different neighborhoods for effective long-range modeling. The proposed G3D module leverages dense cross-spacetime edges as skip connections for direct information propagation across the spatial-temporal graph. By coupling these proposals, we develop a powerful feature extractor named MS-G3D based on which our model outperforms previous state-of-the-art methods on three large-scale datasets: NTU RGB+D 60, NTU RGB+D 120, and Kinetics Skeleton 400.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/58767402/223127347-135bb92b-2dee-46d9-95fc-cebf65c27fc8.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

Assume that you are located at `$MMACTION2/projects/msg3d`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the NTU60 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Data Preparation

Prepare the NTU60 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/skeleton/README.md).

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### NTU60_XSub_2D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  MSG3D   |   92.3   |     10 clips     | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/msg3d/configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230309-73b97296.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |

### NTU60_XSub_3D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  MSG3D   |   89.6   |     10 clips     | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/msg3d/configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20230308-c325d222.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@inproceedings{liu2020disentangling,
  title={Disentangling and unifying graph convolutions for skeleton-based action recognition},
  author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
  booktitle={CVPR},
  pages={143--152},
  year={2020}
}
```
# UMT Project

[Unmasked Teacher: Towards Training-Efficient Video Foundation Models](https://arxiv.org/abs/2303.16058)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video Foundation Models (VFMs) have received limited exploration due to high computational costs and data scarcity. Previous VFMs rely on Image Foundation Models (IFMs), which face challenges in transferring to the video domain. Although VideoMAE has trained a robust ViT from limited data, its low-level reconstruction poses convergence difficulties and conflicts with high-level cross-modal alignment. This paper proposes a training-efficient method for temporal-sensitive VFMs that integrates the benefits of existing methods. To increase data efficiency, we mask out most of the low-semantics video tokens, but selectively align the unmasked tokens with IFM, which serves as the UnMasked Teacher (UMT). By providing semantic guidance, our method enables faster convergence and multimodal friendliness. With a progressive pre-training framework, our model can handle various tasks including scene-related, temporal-related, and complex video-language understanding. Using only public sources for pre-training in 6 days on 32 A100 GPUs, our scratch-built ViT-L/16 achieves state-of-the-art performances on various video tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/58767402/262291190-bdaa6899-e1d6-460f-b329-23d8b38511f3.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

Assume that you are located at `$MMACTION2/projects/umt`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the Kinetics dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/tree/main/tools/data/kinetics#readme).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### Kinetics400

| frame sampling strategy | resolution | backbone |  pretrain   | top1 acc | testing protocol |                             config                              |                             ckpt                              |
| :---------------------: | :--------: | :------: | :---------: | :------: | :--------------: | :-------------------------------------------------------------: | :-----------------------------------------------------------: |
|        uniform 8        |  224x224   |  UMT-B   | Kinetics710 |  87.33   | 4 clips x 3 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/umt/configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.pth) |
|        uniform 8        |  224x224   |  UMT-L   | Kinetics710 |  90.21   | 4 clips x 3 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/umt/configs/umt-large-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-large-p16-res224_kinetics710-pre-ft_u8_k400-rgb/umt-large-p16-res224_kinetics710-pre-ft_u8_k400-rgb.pth) |

### Kinetics700

| frame sampling strategy | resolution | backbone |  pretrain   | top1 acc | testing protocol |                             config                              |                             ckpt                              |
| :---------------------: | :--------: | :------: | :---------: | :------: | :--------------: | :-------------------------------------------------------------: | :-----------------------------------------------------------: |
|        uniform 8        |  224x224   |  UMT-B   | Kinetics710 |  77.95   | 4 clips x 3 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/umt/configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-base-p16-res224_kinetics710-pre-ft_u8_k700-rgb/umt-base-p16-res224_kinetics710-pre-ft_u8_k700-rgb.pth) |
|        uniform 8        |  224x224   |  UMT-L   | Kinetics710 |  82.79   | 4 clips x 3 crop | [config](https://github.com/open-mmlab/mmaction2/blob/main/projects/umt/configs/umt-large-p16-res224_kinetics710-pre-ft_u8_k700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-large-p16-res224_kinetics710-pre-ft_u8_k700-rgb/umt-large-p16-res224_kinetics710-pre-ft_u8_k700-rgb.pth) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@article{li2023unmasked,
  title={Unmasked teacher: Towards training-efficient video foundation models},
  author={Li, Kunchang and Wang, Yali and Li, Yizhuo and Wang, Yi and He, Yinan and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2303.16058},
  year={2023}
}
```
