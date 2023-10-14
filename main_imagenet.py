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
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
import glob

from sklearn import metrics
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from skimage.segmentation import mark_boundaries

from torchvision.io.image import read_image

from sklearn import metrics
import cv2
import random
import warnings
import json
import pandas as pd
warnings.filterwarnings("ignore")

from sam_explainer import *


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.008, help='lr')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--data', type=str, default='imagenet')


args = parser.parse_args()
print('using dataset ',args.data)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
fix_seed(args.seed)

print("using sam")
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)
    

root_dir = '/your/imagenet/root/dir'

str_class = open(os.path.join(root_dir,'LOC_synset_mapping.txt'),'r')

label_to_name = {}
for line in str_class:
    res = line.strip().split(',')[0]
    label_to_name[res[:9]] = res[10:]
    
    
            
gt = json.load(open(root_dir+'imagenet_class_index.json'))
new_gt = {gt[i][0]: int(i) for i in gt}

num_label_to_name = {}

for i in label_to_name:
    num_label_to_name[new_gt[i]] = label_to_name[i]
    



class MyDataset(torch.utils.data.Dataset): ### diy an ImageNet dataset
    def __init__(self, csv_path, images_folder,gt, transform = None):
        self.df = pd.read_csv(csv_path)
        self.df.drop([0])
        self.images_folder = images_folder
        self.transform = transform
        self.gt = gt

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        
        filename = str(self.df.iloc[index,0])
#         label = self.class2index[self.df[index, 1]]
        label = self.gt[self.df.iloc[index,1].split(' ')[0]]
        image_dir = os.path.join(self.images_folder, 'ILSVRC/Data/CLS-LOC/val',filename+'.JPEG')
        image = Image.open(image_dir)
        image = image.convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image_dir

test_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) ### the complete imagenet data pre-process

image_reshape = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
]) ### process the imagenet data to sam


image_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) ### imagenet std and mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.model == 'resnet50':
    model = models.resnet50(weights='IMAGENET1K_V2').to(device)
    cvmodel = model.cuda()
    cvmodel.eval()
    feat_exp = create_feature_extractor(cvmodel, return_nodes=['avgpool'])
    fc = model.fc



model.eval()
feat_exp.eval()
cvmodel.eval()


        

auc_total = 0.0
auc_total_list = []

data_iter = None

img_net = MyDataset(csv_path=root_dir+'LOC_val_solution.csv',images_folder=root_dir,gt=new_gt,transform=test_preprocess)
img_loader = DataLoader(img_net,batch_size=128)

### random select 10000 imagenet sample to explain
num_sample = 10000
idx_path = "imagenet_random/{}_selected.pkl".format(num_sample)
if not os.path.isfile(idx_path):
    print("creating random list "+idx_path)
    data_iter = random.sample(list(range(len(img_net))),num_sample)
    with open(idx_path,'wb') as f:
        pickle.dump(data_iter, f)
else:
    print("loading random list "+idx_path)
    data_iter = pickle.load(open(idx_path,'rb'))

print("explaining {} images".format(num_sample))

num_sample = len(data_iter)


for idx in tqdm(data_iter):
    x , y, img_dir = img_net[idx]
    clean_pil_load_img = Image.open(img_dir).convert('RGB')
    x = test_preprocess(clean_pil_load_img)
    with torch.no_grad():
        soft_org = torch.nn.functional.softmax(cvmodel(x.unsqueeze(0).cuda()),dim=1)
    image_class = int(torch.argmax(soft_org))
    probabilitie_org = float(torch.max(soft_org))
    for_mask_image = np.array(image_reshape(clean_pil_load_img)) ### np int type

    input_image_copy = for_mask_image.copy()
    concept_masks = None
    org_masks = gen_concept_masks(mask_generator,input_image_copy)
    concept_masks = np.array([i['segmentation'].tolist() for i in org_masks])
    auc_mask, shap_list = samshap(cvmodel,input_image_copy,image_class,concept_masks,fc,feat_exp,image_norm=image_norm,lr=args.lr)
    if type(auc_mask) == float:
        auc_total = auc_total + auc_mask
        continue

#     if args.delete:
#         auc_mask = 1 - auc_mask
        
    val_img_numpy = np.expand_dims(input_image_copy,0)
    val_img_numpy = (val_img_numpy * auc_mask).astype(np.uint8)
    batch_img = []
    for i in range(val_img_numpy.shape[0]): 
        batch_img.append(image_norm(val_img_numpy[i,:,:,:]))

    batch_img = torch.stack(batch_img).cuda()

    with torch.no_grad():
        out = torch.nn.functional.softmax(cvmodel(batch_img),dim=1)[:,image_class]
    out = out.cpu().numpy()

    out[out>= probabilitie_org] = probabilitie_org ### norm the upper bound of output to the original acc
    out = out/probabilitie_org
    x_axis = np.array(list(range(1,out.shape[0]+1)))/out.shape[0] *100
    if x_axis.shape[0] == 1:
        auc_tmp = float(out)
    else:
        auc_tmp = float(metrics.auc(x_axis, out))

    auc_total = auc_total + auc_tmp
    
print("this is the mean auc ",auc_total/num_sample)

