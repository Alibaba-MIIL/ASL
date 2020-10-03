# Asymmetric Loss For Multi-Label Classification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymmetric-loss-for-multi-label/multi-label-classification-on-ms-coco)](https://paperswithcode.com/sota/multi-label-classification-on-ms-coco?p=asymmetric-loss-for-multi-label)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymmetric-loss-for-multi-label/multi-label-classification-on-nus-wide)](https://paperswithcode.com/sota/multi-label-classification-on-nus-wide?p=asymmetric-loss-for-multi-label)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymmetric-loss-for-multi-label/multi-label-classification-on-pascal-voc-2007)](https://paperswithcode.com/sota/multi-label-classification-on-pascal-voc-2007?p=asymmetric-loss-for-multi-label)<br>
<br>
[Paper](https://arxiv.org/abs/2009.14119) | [Pretrained models](MODEL_ZOO.md)

Official PyTorch Implementation

> Emanuel Ben-Baruch, Tal Ridnik, Nadav Zamir, Asaf Noy, Itamar
> Friedman, Matan Protter, Lihi Zelnik-Manor<br/> DAMO Academy, Alibaba
> Group

Note - Github page is still under develop. More content soon...


**Abstract**

> Pictures of everyday life are inherently multi-label in nature. Hence,
> multi-label classification is commonly used to analyze their content.
> In typical multi-label datasets, each picture contains only a few
> positive labels, and many negative ones. This positive-negative
> imbalance can result in under-emphasizing gradients from positive
> labels during training, leading to poor accuracy. In this paper,
> we introduce a novel asymmetric loss ("ASL"), that operates
> differently on positive and negative samples. The loss dynamically
> down-weights the importance of easy negative samples, causing the
> optimization process to focus more on the positive samples, and also
> enables to discard mislabeled negative samples. We demonstrate how ASL
> leads to a more "balanced" network, with increased average
> probabilities for positive samples, and show how this balanced network
> is translated to better mAP scores, compared to commonly used losses.
> Furthermore, we offer a method that can dynamically adjust the level
> of asymmetry throughout the training. With ASL, we reach new
> state-of-the-art results on three common multi-label datasets,
> including achieving $86.6\%$ on MS-COCO. We also demonstrate ASL
> applicability for other tasks such as fine-grain single-label
> classification and object detection. ASL is effective, easy to
> implement, and does not increase the training time or complexity

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/ASL_comparison.png" align="center" width="400" ></td>
  </tr>
</table>
</p>

## Asymmetric Loss (ASL) Implementation
In this PyTorch [file](\src\loss_functions\losses.py), we provide two
implementations of our new loss function, ASL, that can serve as a
drop-in replacement for standard loss functions (Cross-Entropy and
Focal-Loss)

The two implementations are: 
- ```class AsymmetricLoss(nn.Module)```
- ```class AsymmetricLossOptimized(nn.Module)``` <br>

The two losses are bit-accurate. However, ``` AsymmetricLossOptimized```
contains a more optimized (and complicated) way of implementing ASL,
which minimizes memory allocations, gpu uploading, and favors inplace
operations.

## Pretrained Models
In this [link](MODEL_ZOO.md), we provide pre-trained models on various
dataset. 

## Inference Code
We provide [inference code](infer.py), that demonstrate how to load our
model, pre-process an image and do actuall inference. This code can also
serve as a template for validating our mAP scores. Example run of
MS-COCO model (after downloading the relevant model):
```
python infer.py  \
--dataset_type=MS-COCO \
--model_name=tresnet_l \
--model_path=./models_local/MS_COCO_TRresNet_L_448_86.6.pth \
--pic_path=./pics/000000000885.jpg \
--input_size=448
```
which will result in:
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/example_inference.jpeg" align="center" width="600" ></td>
  </tr>
</table>
</p>

Example run of OpenImages model:
```
python infer.py  \
--dataset_type=OpenImages \
--model_name=tresnet_l \
--model_path=./models_local/Open_ImagesV6_TRresNet_L_448.pth \
--pic_path=./pics/000000000885.jpg \
--input_size=448
```
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/example_inference_open_images.jpeg" align="center" width="600" ></td>
  </tr>
</table>
</p>

## Citation
```
 @misc{benbaruch2020asymmetric, 
        title={Asymmetric Loss For Multi-Label Classification}, 
        author={manuel Ben-Baruch and Tal Ridnik and Nadav Zamir and Asaf Noy and Itamar Friedman and Matan Protter and Lihi Zelnik-Manor}, 
        year={2020}, 
        eprint={2009.14119},
        archivePrefix={arXiv}, 
        primaryClass={cs.CV} }
```

## Contact
Feel free to contact if there are any questions or issues - Emanuel
Ben-Baruch (emanuel.benbaruch@alibaba-inc.com) or Tal Ridnik (tal.ridnik@alibaba-inc.com).
