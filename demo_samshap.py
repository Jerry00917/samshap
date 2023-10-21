from __future__ import print_function, division
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import glob
from utils.shap_utils import *
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
from sam_explainer import *



### define a tagret model
model = models.resnet50(weights='IMAGENET1K_V2').cuda()
cvmodel = model.cuda()
cvmodel.eval()
feat_exp = create_feature_extractor(cvmodel, return_nodes=['avgpool'])
fc = model.fc
model.eval()
feat_exp.eval()
cvmodel.eval()


### define a sam
sam = sam_model_registry["default"](checkpoint="/your_sam_dir/sam_vit_h_4b8939.pth")
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)


### define three wats to pre-process a given image
test_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) ### a complete imagenet data pre-process

image_reshape = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
]) ### process the imagenet data to sam


image_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) ### imagenet std and mean


### a test image (a hummingbird)
img = Image.open('hum2.jpg').convert('RGB')



predict_org = torch.nn.functional.softmax(model(test_preprocess(img).unsqueeze(0).cuda()),dim=1)
pred_image_class = int(torch.argmax(predict_org))

for_mask_image = np.array(image_reshape(img)) ### np int type

input_image_copy = for_mask_image.copy()
org_masks = gen_concept_masks(mask_generator,input_image_copy)
concept_masks = np.array([i['segmentation'].tolist() for i in org_masks])
auc_mask, shap_list = samshap(model,input_image_copy,pred_image_class,concept_masks,fc,feat_exp,image_norm=image_norm)


### select the concept patch with the highest shapley value
final_explain = (for_mask_image*auc_mask[0])

final_explain = Image.fromarray(final_explain)
final_explain.save("your_final_explain_output.jpeg")






















