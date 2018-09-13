import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from loaddata import kittidata, kittidata_split

import numpy as np
import time
import os, shutil
import argparse

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='pytorch FCN on Kitti')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='the training batch size(default: 8)')
parser.add_argument('--split', type=float, default=0.8, metavar='Split',
                    help='the split ratio indicates how much percentage of train set, '
                         'the rest will for validation (default: 0.8)')
parser.add_argument('--resize-ratio', type=float, default=0.6, metavar='Resize',
                    help='how much ratio to resize(shrink) the training image(default: 0.6)')
parser.add_argument('--numloader', type=int, default=8, metavar='Nl',
                    help='the num of CPU for data loading. 0 means only use one CPU. (default: 8)')
parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='the required total training epochs(default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='Lr',
                    help='the learning rate for decoder part(default: 1e-3)')
parser.add_argument('--lr-pretrain', type=float, default=1e-4, metavar='LR-P',
                    help='the learning rate for encoder(pre-trained) part(default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='the momentum of optimizer(default: 0)')
parser.add_argument('--decay', type=float, default=1e-5, metavar='D',
                    help='the weight for L2 regularization(default: 1e-5)')
parser.add_argument('--step', type=int, default=10, metavar='Step',
                    help='learning rate will decay after the step(default: 10)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='Gamma',
                    help='learning rate will decay gamma percent after few steps(default: 0.5)')
parser.add_argument('--logdir', type=str, default='log', metavar='Log',
                    help='the folder to store the tensorboard logs(default: log)')
parser.add_argument('--vgg', type=str, default='19', metavar='Vgg',
                    help='the configuration of vgg(default: 19)')
parser.add_argument('--fcn', type=str, default='1', choices=['1', '8', '16', '32'], metavar='FCN',
                    help='the configuration of FCN(default: 1)')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], metavar='Mode',
                    help='train or test(default: train)')
parser.add_argument('--model', type=str, default='', metavar='Model',
                    help='the pre-trained model or the model for test. if not specify, it will initial '
                         'a new model (default: '')')

args = parser.parse_args()

# remove log if it already existed
if args.logdir not in os.listdir('./'):
    os.mkdir('./'+args.logdir)
else:
    shutil.rmtree('./'+args.logdir)
    os.mkdir('./' + args.logdir)
writer = SummaryWriter(args.logdir)

n_class = 12

batch_size = args.batch_size
split_rate = args.split
resize_ratio = args.resize_ratio
multi_thread_loader = args.numloader
epochs = args.epochs
lr = args.lr
lr_pretrain = args.lr_pretrain
momentum = args.momentum
w_decay = args.decay
step_size = args.step
gamma = args.gamma
vgg_config = 'vgg' + args.vgg

# the config format likes below
# vgg-config_fcn-config_criterion_batch-size_epoch_optimizer_step_gamma_lr_lr-pretrain-momentum-w_decay
configs = "vgg{}_fcn{}-BCEWithLogits_{}_{}_RMSprop_{}_{}" \
          "_{}_{}_{}".format(args.vgg, args.fcn, batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

#train_depth_path = './kitti_semseg_unizg/train/depth'
train_rgb_path = './kitti_semseg_unizg/train/rgb'
train_label_path = './kitti_semseg_unizg/train/labels'

#test_depth_path = './kitti_semseg_unizg/test/depth'
test_rgb_path = './kitti_semseg_unizg/test/rgb'
test_label_path = './kitti_semseg_unizg/test/labels'

kittiset = kittidata_split(train_rgb_path, train_label_path, split_rate)

train_data, train_label = kittiset.getdata('train')
val_data, val_label = kittiset.getdata('val')

train_set = kittidata(train_data, train_label, shrink_rate=resize_ratio, flip_rate=0.5)
val_set = kittidata(val_data, val_label, shrink_rate=1, flip_rate=0)  # keep data unchanged
print('{} for training, {} for validation'.format(len(train_data), len(val_data)))
print('data loading finished')

# set the validation batch_size equal to the device amount to fully utilize (multi)GPUs
device_amount = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
print('there are {} devices'.format(device_amount))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=multi_thread_loader)
val_loader = DataLoader(val_set, batch_size=device_amount, shuffle=False, num_workers=multi_thread_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model:
    fcn_model = torch.load(args.model).to(device)
else:
    vgg_model = VGGNet(requires_grad=True, model=vgg_config, remove_fc=True).to(device)
    fcn_model = nn.DataParallel(FCNs(pretrained_net=vgg_model, n_class=n_class)).to(device)

criterion = nn.NLLLoss()

params = list()
for name, param in fcn_model.named_parameters():
    if 'pretrained_net' in name:  # use small learning rate for
        params += [{'params': param, 'lr': lr_pretrain}]
    else:
        params += [{'params': param, 'lr': lr}]

optimizer = optim.RMSprop(params, weight_decay=w_decay)
optimizer = nn.DataParallel(optimizer)
scheduler = lr_scheduler.StepLR(optimizer.module, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5

total_train_set = len(train_data)
total_iter = len(train_loader)

def train():
    ti = time.time()
    for epoch in range(epochs):
        scheduler.step()
        epoch_loss = 0
        ts = time.time()
        for iter, (inputs, labels, num_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, num_labels = inputs.to(device), num_labels.to(device)
            outputs = fcn_model(inputs)
            loss = criterion(F.log_softmax(outputs), num_labels).to(device)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.module.step()

            writer.add_scalar('single_iter_loss', loss.item(), epoch * total_iter + iter)
            if iter % 10 == 0:
                print("epoch{}, iter{}, time elapsed {}, loss {}".format(epoch, iter,
                                                                        timeformat(int(time.time()-ti)), loss.item()))
        mel = epoch_loss / total_iter
        writer.add_scalar('mean_epoch_loss', mel, epoch)
        print("Epoch {} takes {} with mean loss {:.5f}".format(epoch, timeformat(int(time.time()-ts)), mel))
        torch.save(fcn_model, model_path + '_epoch{}'.format(epoch))  # store the model each epoch

        val(epoch)


def val(epoch):
    print('validation starts')
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    tol_time = 0
    with torch.no_grad():
        for iter, (inputs, labels, num_labels) in enumerate(val_loader):
            ti = time.time()
            inputs = inputs.to(device)
            output = fcn_model(inputs)
            output = output.data.cpu().numpy()

            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            if iter == 0:
                for i in range(device_amount):
                    writer.add_image('result of {}'.format(i),
                                     image_grid(inputs[i].unsqueeze(0), torch.from_numpy(pred[i]).unsqueeze(0), num_labels[i].unsqueeze(0)), epoch)

            target = num_labels.numpy()
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))

            ti = time.time() - ti
            tol_time += ti

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    writer.add_scalar('pixel_acc', pixel_accs, epoch)
    writer.add_scalar('meanIoU', np.nanmean(ious), epoch)

    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    print('total validation time is {}'.format(timeformat(int(tol_time))))


def test():
    kittiset_test = kittidata_split(test_rgb_path, test_label_path, 0)
    test_data, test_label = kittiset_test.getdata('val')
    test_set = kittidata(test_data, test_label, shrink_rate=1, flip_rate=0)
    test_loader = DataLoader(test_set, batch_size=device_amount, shuffle=False, num_workers=multi_thread_loader)

    print('Finishded loading test data. Test starts!')

    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    tol_time = 0
    with torch.no_grad():
        for iter, (inputs, labels, num_labels) in enumerate(test_loader):
            ti = time.time()
            inputs = inputs.to(device)
            output = fcn_model(inputs)
            output = output.data.cpu().numpy()

            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            target = num_labels.numpy()
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))

            ti = time.time() - ti
            tol_time += ti

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()

    print("pix_acc: {}, meanIoU: {}, IoUs: {}".format(pixel_accs, np.nanmean(ious), ious))
    print('total test time is {}'.format(timeformat(int(tol_time))))

# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
            # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


# align images (batch, orginial, pred, ground_truth)
def image_grid(image, pred, label):
    l = list()
    for i in range(image.shape[0]):
        l.extend([train_set.denormalize(image[i]), train_set.visualize(pred[i]), train_set.visualize(label[i])])
    return utils.make_grid(l, nrow=3, padding=20)

def timeformat(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    t = '{:>02d}:{:>02d}:{:>02d}'.format(h, m, s) if h else '{:>02d}:{:>02d}'.format(m, s) if m else '{:2d}s'.format(s)
    return t

if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    if args.mode == 'train':
        train()
    else:
        test()