import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)
import numpy as np
import fire
import os.path as osp
from detectron2.config import get_cfg
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from tools.train_net import Trainer, DetectionCheckpointer
from glob import glob

import torchvision as tv
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

import matplotlib.colors
import seaborn as sns
import torchvision.ops as ops
from torchvision.ops import box_area, box_iou
import random

import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont
from copy import copy
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget

clipart_classes = ['sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse']

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    instances = outputs[0]['instances']
    pred_labels = instances.pred_classes.cpu().numpy()
    pred_scores = instances.scores.detach().cpu().numpy()
    pred_bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()
    pred_classes = [clipart_classes[i] for i in instances.pred_classes.cpu().numpy()]
 
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

def main(
        config_file="configs/fish/vitl_shot10_fish_finetune.yaml", 
        rpn_config_file="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml",
        # model_path="weights/trained/open-vocabulary/lvis/vitl_0069999.pth", 
        model_path="output/vitl/top_fish_10shot/all_25.2/model_final.pth", 
        image_dir='tools/input', 
        output_dir='cam/vitl/fish_10shot', 
        device='cuda:0',
        overlapping_mode=True,
        topk=1,
        output_pth=False,
        threshold=0.45
    ):
    assert osp.abspath(image_dir) != osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
    config.DE.TOPK = topk
    config.MODEL.MASK_ON = False
    print(config)
    # config.DE.CLASS_PROTOTYPES = "prototypes_init/clipart1k_10shot.vitl14.bbox.p10.sk.pkl"
    # config.DE.BG_PROTOTYPES = "weights/initial/background/background_prototypes.vitl14.pth"

    config.freeze()
    
    augs = utils.build_augmentation(config, False)
    augmentations = T.AugmentationList(augs) 

    # building models
    model = Trainer.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()
    model = model.to(device)
    print("model.offline_backbone.res4:-----------",model.offline_backbone.res4)

    print(glob(osp.join(image_dir, '*')))
    for img_file in glob(osp.join(image_dir, '*')):
        base_filename = osp.splitext(osp.basename(img_file))[0]
        print(base_filename)

        dataset_dict = {}
        image = utils.read_image(img_file, format="RGB")
        dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]

        aug_input = T.AugInput(image)
        augmentations(aug_input)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(device)

        batched_inputs = dataset_dict["image"].unsqueeze(0)

        from pytorch_grad_cam import GradCAM, EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import cv2

        boxes, classes, labels, indices = predict(batched_inputs, model, device, 0.01)
        targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
        with EigenCAM(model=model, target_layers=[model.offline_backbone.res4]) as cam:
        # with GradCAM(model=model, target_layers=[model.per_cls_cnn.main_layers[-1]]) as cam:
            cam.batch_size = 1
            grayscale_cam = cam(input_tensor=batched_inputs, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            image_resize = cv2.resize(image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
            cam_image = show_cam_on_image(image_resize/255, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            os.makedirs(output_dir, exist_ok=True)
            cam_output_path = os.path.join(output_dir, f'{base_filename}_cam_ours2.jpg')
            cv2.imwrite(cam_output_path, cam_image) 

            print("yes!!!!!!!")


if __name__ == "__main__":
    main()