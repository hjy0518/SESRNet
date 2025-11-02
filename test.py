import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn.functional as F

import numpy as np

import time
from model.SESRNet import SESRNet
from data1 import test_dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './ORSI/Test/'

model = SESRNet()
model.load_state_dict(torch.load('./cpts/MyNet_best_ORS4199.pth'))
model.cuda()
model.eval()

# test_datasets = ['EORSSD']
# test_datasets = ['ORSSD']
test_datasets = ['ORS-4199']
for dataset in test_datasets:

    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/gt/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    mae_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        res,s1_sig = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res*255)
        print(name + "  finish!")
        mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

    mae = mae_sum / test_loader.size
    print(dataset,'Res mae is : ',mae)