import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from model.SESRNet import SESRNet
from data1 import get_loader
from data1 import test_dataset
from utils.utils import clip_gradient, adjust_lr
import pytorch_iou

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
# % For vgg, batchsize is 6; for ResNet, batchsize is 8.
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models

model = KPNet()
load = './model/smt_tiny.pth'
# if load is not None:
#     model.load_pre(load)
#     print('load model from',load)
# model.load_state_dict(torch.load('./cpts/MyNet_best_ORS4199.pth'))

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# image_root = './ORSI/Train/EORSSD/Images/'
# gt_root = './ORSI/Train/EORSSD/gt/'
#
# test_image_root = './ORSI/Test/EORSSD/Images/'
# test_gt_root = './ORSI/Test/EORSSD/gt/'



# image_root = './ORSI/Train/ORSSD/Images/'
# gt_root = './ORSI/Train/ORSSD/gt/'
#
# test_image_root = './ORSI/Test/ORSSD/Images/'
# test_gt_root = './ORSI/Test/ORSSD/gt/'

image_root = './ORSI/Train/ORS-4199/Images/'
gt_root = './ORSI/Train/ORS-4199/gt/'
test_image_root = './ORSI/Test/ORS-4199/Images/'
test_gt_root = './ORSI/Test/ORS-4199/gt/'

save_path = './cpts/'


train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
MSE = torch.nn.MSELoss()
def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        s1, s1_sig = model(images)

        loss1 = CE(s1, gts) + IOU(s1_sig, gts)

        loss = loss1

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 500 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss_ce: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + 'MyNet_{}.pth'.format(epoch))

best_mae=1
best_epoch=0
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res, s1_sig= model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))

        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'MyNet_epoch_best.pth')
            print('best epoch:{}'.format(epoch))



print("Let's go!")
for epoch in range(1,opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
    test(test_loader,model,epoch,save_path)
