import torch
from util.box_utils import nms, chamferDist, to_var
from data.oiltankLoader import VOCDetection as OilDataset
from torch.utils.data import DataLoader
from models.skeleton import NNSkeleton
from models.resnet import FPN, resnet34

""" Parameters here
"""
is_cuda = True
dataroot = '/home/x/data/VOCdevkit'
num_epochs = 100

ds = OilDataset(dataroot, image_sets=[('2007', 'trainval')])
dl = DataLoader(ds, batch_size=1)

net = NNSkeleton(resnet34(True))
if is_cuda:
    net.cuda()

for epoch in range(num_epochs):
    for ii, minibatch in enumerate(dl, 0):
        xs, ts = to_var(minibatch[0]), to_var(minibatch[1])
        # print(xs)
        bboxes, scores = net(xs)
        # print(bboxes.size())
        # print(scores.size())
        keep, count = nms(bboxes.data, scores.data)
        print(keep, count)

